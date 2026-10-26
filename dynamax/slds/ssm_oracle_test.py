"""
Equivalence tests against ``lindermanlab/ssm`` (commit eb6c8aa) on the deterministic sub-steps of Laplace
variational EM: Hessian blocks, the Newton mode, the information-form smoother, the q(z) update, the dynamics and
emission M-steps and the recurrent M-step objective. They read ``dynamax/slds/fixtures/ssm_fixtures.npz``,
regenerated locally with ``fixtures/export_ssm_fixtures.py`` (see ``fixtures/README.md``), and skip when it is
absent; ``DYNAMAX_SSM_FIXTURES`` overrides the path. The comparisons need 64-bit floats: ``test_oracle_under_x64``
re-runs this file in a subprocess with ``JAX_ENABLE_X64=1``.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.linear_gaussian_ssm.block_tridiag import BlockTridiagPotentials, _sample_with_noise, block_tridiag_mvn_smoother
from dynamax.slds import GaussianSLDS, ParamsSLDS, RecurrentGaussianSLDS
from dynamax.slds.inference import (
    continuous_posterior_from_potentials, discrete_entropy, laplace_newton, update_continuous_posterior,
    update_discrete_posterior)
from dynamax.slds.transitions import ParamsRecurrentSLDSTransitions, ParamsStationarySLDSTransitions

X64 = jax.config.read("jax_enable_x64")
FIXTURE = os.environ.get("DYNAMAX_SSM_FIXTURES", os.path.join(os.path.dirname(__file__), "fixtures", "ssm_fixtures.npz"))
CASES = [f"{t}/{s}/{r}" for t in ["standard", "recurrent", "recurrent_only"] for s in ["single", "multi"] for r in ["default", "noreg"]]

pytestmark = pytest.mark.skipif(not os.path.exists(FIXTURE), reason=f"ssm fixture {FIXTURE} not found (see fixtures/README.md)")


@pytest.fixture(scope="module")
def fixtures():
    data = np.load(FIXTURE)
    return {key: data[key] for key in data.files}


class Case:
    """A dynamax model and parameters that mirror one exported ssm configuration."""
    K, D, N = 3, 2, 8
    ZERO_DYNAMICS_REG = dict(l2_penalty_weights=0.0, l2_penalty_input_weights=0.0, l2_penalty_bias=0.0, nu0=0.0, Psi0=0.0)

    def __init__(self, name, fx):
        self.name = name
        self.get = lambda key: jnp.asarray(fx[f"{name}/{key}"])
        self.has = lambda key: f"{name}/{key}" in fx
        transitions, subspace, reg = name.split("/")
        single = subspace == "single"
        dynamics_kwargs = self.ZERO_DYNAMICS_REG if reg == "noreg" else None
        kwargs = dict(single_subspace=single, dynamics_cov_type="full", emission_cov_type="diag", dynamics_kwargs=dynamics_kwargs, damping=0.0)
        if transitions == "standard":
            self.model = GaussianSLDS(self.K, self.D, self.N, **kwargs)
            trans_params = ParamsStationarySLDSTransitions(transition_matrix=self.get("trans/transition_matrix"))
        else:
            self.model = RecurrentGaussianSLDS(self.K, self.D, self.N, recurrent_only=transitions == "recurrent_only", **kwargs)
            logits = self.get("trans/r")[None] if transitions == "recurrent_only" else self.get("trans/log_Ps")
            trans_params = ParamsRecurrentSLDSTransitions(logits=logits, recurrent_weights=self.get("trans/Rs"), input_weights=self.get("trans/Ws"))
        params, self.props = self.model.initialize(jr.PRNGKey(0))
        self.params = ParamsSLDS(
            initial=params.initial._replace(probs=self.get("pi0")),
            transitions=trans_params,
            dynamics=params.dynamics._replace(initial_mean=self.get("dyn/mu_init")[0], initial_cov=self.get("dyn/Sigmas_init")[0],
                                              weights=self.get("dyn/As"), biases=self.get("dyn/bs"), input_weights=self.get("dyn/Vs"),
                                              covs=self.get("dyn/Sigmas")),
            emissions=params.emissions._replace(weights=self.get("emis/Cs"), biases=self.get("emis/ds"), input_weights=self.get("emis/Fs"),
                                                covs=jnp.exp(self.get("emis/inv_etas"))))
        self.y = self.get("y")
        self.T = self.y.shape[0]
        self.inputs = jnp.zeros((self.T, 0))
        self.Ez, self.Ezzp1 = self.get("qz/Ez"), self.get("qz/Ezzp1")
        self.Ps = self.get("qz/Ps")
        if transitions == "standard":
            # ssm stores one (1, K, K) matrix and the time-summed pairwise expectations for stationary transitions
            self.Ezzp1 = self.Ezzp1.sum(0) if self.Ezzp1.ndim == 3 else self.Ezzp1
            self.Ps = self.Ps[0] if self.Ps.ndim == 3 else self.Ps

    def potentials(self, prefix):
        """ssm's (J_ini, J_dyn_11/21/22, J_obs, h_*) as dynamax block-tridiagonal potentials."""
        g = lambda key: self.get(f"{prefix}/{key}")
        J_diag = g("J_obs").at[0].add(g("J_ini")).at[:-1].add(g("J_dyn_11")).at[1:].add(g("J_dyn_22"))
        h = g("h_obs").at[0].add(g("h_ini")).at[:-1].add(g("h_dyn_1")).at[1:].add(g("h_dyn_2"))
        return BlockTridiagPotentials(J_diag=J_diag, J_lower=g("J_dyn_21"), h=h)

    def zero_covariance_posterior(self, x):
        """A q(x) concentrated at x, under which the exact expectations are ssm's plug-in values."""
        T, D = x.shape
        qx = continuous_posterior_from_potentials(BlockTridiagPotentials(jnp.tile(jnp.eye(D), (T, 1, 1)), jnp.zeros((T - 1, D, D)), x))
        return qx._replace(smoothed_means=x, smoothed_covariances=jnp.zeros((T, D, D)),
                           smoothed_cross_covariances=jnp.einsum("ti,tj->tij", x[:-1], x[1:]))


@pytest.mark.skipif(not X64, reason="needs 64-bit floats; run through test_oracle_under_x64")
@pytest.mark.parametrize("name", CASES)
class TestOracle:
    def test_hessian_and_linear_term(self, name, fixtures):
        """Negative Hessian blocks and h = J x at the x sample match ssm's."""
        case = Case(name, fixtures)
        trans, dyn, emis = case.model._components
        x = case.get("x_sample")
        J_diag, J_lower = dyn.neg_hessian_blocks(case.params.dynamics, case.Ez, case.inputs)
        J_diag = J_diag + emis.neg_hessian_blocks(case.params.emissions, case.y, x, case.Ez, case.inputs)
        if trans.depends_on_x:
            J_diag = J_diag.at[:-1].add(trans.neg_hessian_x(case.params.transitions, x, case.inputs, case.Ezzp1))
        reference = case.potentials("hess")
        assert jnp.allclose(J_diag, reference.J_diag, atol=1e-8, rtol=1e-8)
        assert jnp.allclose(J_lower, reference.J_lower, atol=1e-8, rtol=1e-8)
        from dynamax.linear_gaussian_ssm.block_tridiag import block_tridiag_mvn_matvec
        assert jnp.allclose(block_tridiag_mvn_matvec(J_diag, J_lower, x), reference.h, atol=1e-8, rtol=1e-8)

    def test_expected_log_joint(self, name, fixtures):
        """The expected log joint at the x sample equals ssm's (whose x-dependent terms are plug-in values)."""
        case = Case(name, fixtures)
        trans, dyn, emis = case.model._components
        x = case.get("x_sample")
        value = (jnp.sum(case.Ez[0] * jnp.log(case.params.initial.probs))
                 + trans.expected_log_trans_prob(case.params.transitions, x, case.inputs, case.Ezzp1)
                 + dyn.expected_log_joint_x(case.params.dynamics, x, case.Ez, case.inputs)
                 + emis.expected_log_joint_x(case.params.emissions, case.y, x, case.Ez, case.inputs))
        assert jnp.allclose(-value, case.get("hess/neg_expected_log_joint"), rtol=1e-8)

    def test_newton_mode_and_laplace_precision(self, name, fixtures):
        """From ssm's starting mean and (Ez, Ezzp1), the Newton update lands on ssm's mode with ssm's precision."""
        case = Case(name, fixtures)
        model = case.model
        model.newton_tolerance, model.newton_maxiter = 1e-10, 200
        qz = update_discrete_posterior(model, case.params, case.zero_covariance_posterior(case.get("x_sample")),
                                       case.get("x_sample")[None], case.y, case.inputs)
        qz = qz._replace(smoothed_probs=case.Ez, trans_probs=case.Ezzp1)
        qx0 = continuous_posterior_from_potentials(case.potentials("q0"))
        assert jnp.allclose(qx0.smoothed_means, case.get("x0"), atol=1e-8)
        qx = update_continuous_posterior(model, case.params, qz, qx0, case.y, case.inputs)
        assert jnp.allclose(qx.smoothed_means, case.get("x_star"), atol=1e-5)
        reference = case.potentials("q1")
        assert jnp.allclose(qx.J_diag, reference.J_diag, atol=1e-5, rtol=1e-5)
        assert jnp.allclose(qx.J_lower, reference.J_lower, atol=1e-5, rtol=1e-5)
        assert jnp.allclose(qx.h, reference.h, atol=1e-5, rtol=1e-5)

    def test_smoother_and_entropy(self, name, fixtures):
        """On ssm's potentials the block-tridiagonal smoother returns ssm's moments; the entropies agree."""
        case = Case(name, fixtures)
        post = block_tridiag_mvn_smoother(case.potentials("q1"))
        assert jnp.allclose(post.smoothed_means, case.get("q1/Ex"), atol=1e-8)
        assert jnp.allclose(post.smoothed_covariances, case.get("q1/Sigmas"), atol=1e-8)
        # both store the uncentered E[x_t x_{t+1}^T]
        assert jnp.allclose(post.smoothed_cross_covariances, case.get("q1/ExxnT"), atol=1e-8)
        assert jnp.allclose(post.log_normalizer, case.get("q1/log_Z"), rtol=1e-8)
        # Entropies. ssm guards its logs with +1e-16, which biases its discrete entropy when a pairwise expectation
        # sits on a transition probability far below 1e-16 (by 130 nats in one exported configuration); dynamax's
        # xlogy terms are exact. So: the dynamax discrete entropy equals the exact formula on ssm's own potentials,
        # and the continuous entropy equals ssm's total minus ssm's (guarded) discrete part.
        from jax.scipy.special import xlogy
        from dynamax.linear_gaussian_ssm.block_tridiag import block_tridiag_mvn_entropy
        pi0, ll, log_Z = case.get("qz/pi0"), case.get("qz/log_likes"), case.get("qz/log_Z")
        exact_discrete = log_Z - jnp.sum(xlogy(case.Ez[0], pi0)) - jnp.sum(case.Ez * ll) - jnp.sum(xlogy(case.Ezzp1, case.Ps))
        ssm_discrete = (log_Z - jnp.sum(case.Ez[0] * jnp.log(pi0 + 1e-16)) - jnp.sum(case.Ez * ll)
                        - jnp.sum(case.Ezzp1 * jnp.log(case.Ps + 1e-16)))
        assert jnp.allclose(discrete_entropy(update_discrete_posterior_from_fixture(case)), exact_discrete, rtol=1e-6, atol=1e-6)
        continuous = block_tridiag_mvn_entropy(post.log_det_precision, case.T, case.D)
        assert jnp.allclose(continuous, case.get("q1/entropy") - ssm_discrete, rtol=1e-8)

    def test_sampler_with_ssm_noise(self, name, fixtures):
        """Feeding ssm's N(0, I) draws to the backward sampler reproduces ssm's q(x) sample."""
        case = Case(name, fixtures)
        potentials = case.potentials("q0")
        sample = _sample_with_noise(potentials, case.get("x_sample_noise"))
        assert jnp.allclose(sample, case.get("x_sample"), atol=1e-8)

    def test_discrete_update(self, name, fixtures):
        """The q(z) update from ssm's x sample (as a zero-covariance q(x)) builds ssm's chain potentials, and the
        log-space smoother returns ssm's expectations and log normalizer on them."""
        case = Case(name, fixtures)
        qz = update_discrete_posterior_from_fixture(case)
        assert jnp.allclose(qz.log_likelihoods, case.get("qz/log_likes"), atol=1e-6, rtol=1e-6)
        assert jnp.allclose(jnp.exp(qz.log_transition_matrices), case.Ps, atol=1e-8)
        assert jnp.allclose(qz.log_normalizer, case.get("qz/log_Z"), rtol=1e-8)
        assert jnp.allclose(qz.smoothed_probs, case.get("qz/Ez"), atol=1e-6)
        assert jnp.allclose(qz.trans_probs, case.Ezzp1, atol=1e-6)

    def test_dynamics_m_step(self, name, fixtures):
        """The dynamics M-step on the exact expectations equals ssm's (Q up to the ridge term, ~1e-10)."""
        case = Case(name, fixtures)
        if not case.has("mstep/As"):
            pytest.skip("ssm's unregularized M-step was singular for this configuration")
        dyn = case.model.dynamics_component
        qx = continuous_posterior_from_potentials(case.potentials("q1"))
        stats = jax.tree_util.tree_map(lambda a: a[None], dyn.collect_suff_stats(case.params.dynamics, case.Ez, qx, case.inputs))
        new, _ = dyn.m_step(case.params.dynamics, case.props.dynamics, stats, None)
        assert jnp.allclose(new.weights, case.get("mstep/As"), atol=1e-7, rtol=1e-7)
        assert jnp.allclose(new.biases, case.get("mstep/bs"), atol=1e-7, rtol=1e-7)
        assert jnp.allclose(new.covs, case.get("mstep/Sigmas"), atol=1e-6, rtol=1e-6)

    def test_emission_m_step(self, name, fixtures):
        """The emission M-step with the x sample as a zero-covariance q(x) equals ssm's sample-based update."""
        case = Case(name, fixtures)
        emis = case.model.emission_component
        qx = case.zero_covariance_posterior(case.get("x_sample"))
        stats = jax.tree_util.tree_map(lambda a: a[None], emis.collect_suff_stats(case.params.emissions, case.y, case.Ez, qx, case.inputs))
        new, _ = emis.m_step(case.params.emissions, case.props.emissions, stats, None)
        assert jnp.allclose(new.weights, case.get("mstep/Cs"), atol=1e-7, rtol=1e-7)
        assert jnp.allclose(new.biases, case.get("mstep/ds"), atol=1e-7, rtol=1e-7)
        assert jnp.allclose(new.covs, jnp.exp(case.get("mstep/inv_etas")), atol=1e-7, rtol=1e-7)

    def test_recurrent_m_step_objective(self, name, fixtures):
        """ssm's optimum evaluates to the exported objective value, and dynamax's L-BFGS optimum is at least as good."""
        case = Case(name, fixtures)
        if not case.has("mstep/trans/expected_log_trans"):
            pytest.skip("stationary transitions have a closed-form M-step")
        trans = case.model.transition_component
        x = case.get("x_sample")
        objective = lambda p: trans.log_prior(p) + trans.expected_log_trans_prob(p, x, case.inputs, case.Ezzp1)
        if case.name.startswith("recurrent_only"):
            ssm_opt = ParamsRecurrentSLDSTransitions(logits=case.get("mstep/trans/r")[None], recurrent_weights=case.get("mstep/trans/Rs"), input_weights=case.get("mstep/trans/Ws"))
        else:
            ssm_opt = ParamsRecurrentSLDSTransitions(logits=case.get("mstep/trans/log_Ps"), recurrent_weights=case.get("mstep/trans/Rs"), input_weights=case.get("mstep/trans/Ws"))
        # the same function of the parameters (ssm's Dirichlet log prior drops the normalizing constant, so the priors differ by one)
        assert jnp.allclose(trans.expected_log_trans_prob(ssm_opt, x, case.inputs, case.Ezzp1), case.get("mstep/trans/expected_log_trans"), rtol=1e-6)
        stats = jax.tree_util.tree_map(lambda a: a[None], trans.collect_suff_stats(case.params.transitions, type("Q", (), {"trans_probs": case.Ezzp1})(), x[None], case.inputs))
        ours, _ = trans.m_step(case.params.transitions, case.props.transitions, stats, None)
        # dynamax's optimum is at least as good as ssm's on the convex objective (ssm's scipy L-BFGS stops at a
        # looser tolerance, so its value can be lower by a few units)
        assert objective(ours) >= objective(ssm_opt) - 1e-6 * jnp.abs(objective(ssm_opt))


def update_discrete_posterior_from_fixture(case):
    """The q(z) update with ssm's x sample fed as a zero-covariance q(x) (S = 1, so the normalizer fold is a no-op)."""
    x = case.get("x_sample")
    return update_discrete_posterior(case.model, case.params, case.zero_covariance_posterior(x), x[None], case.y, case.inputs)


def test_oracle_under_x64():
    """Run the oracle comparisons in a 64-bit subprocess."""
    if X64:
        pytest.skip("already running under x64")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "TestOracle"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-4000:]
