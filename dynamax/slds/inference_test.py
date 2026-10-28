"""
Tests for the SLDS variational inference. ``test_exactness_under_x64`` re-runs this file in a 64-bit
subprocess with the tight tolerances.
"""
import itertools
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from dynamax.parameters import ParameterProperties
from dynamax.linear_gaussian_ssm import LinearGaussianSSM, lgssm_filter
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions
from dynamax.slds import GaussianSLDS
from dynamax.slds.inference import backtracking_line_search, log_space_hmm_smoother

X64 = jax.config.read("jax_enable_x64")
TOL = 1e-7 if X64 else 2e-3


def mle_kwargs(state_dim, emission_dim):
    """Hyperparameters that turn every prior off (so that the M-step is maximum likelihood)."""
    return dict(dynamics_kwargs=dict(l2_penalty_weights=0.0, l2_penalty_input_weights=0.0, l2_penalty_biases=0.0,
                                     nu0=-(state_dim + 1), Psi0=0.0),
                emission_kwargs=dict(l2_penalty=0.0, nu0=-(emission_dim + 1), Psi0=0.0))


def to_lgssm(params, state=0):
    """The LGSSM parameters of a single-subspace SLDS given its discrete state (one state, or one per time step)."""
    dyn, emis = params.dynamics, params.emissions
    return ParamsLGSSM(
        initial=ParamsLGSSMInitial(mean=dyn.initial_mean, cov=dyn.initial_cov),
        dynamics=ParamsLGSSMDynamics(weights=dyn.weights[state], bias=dyn.biases[state],
                                     input_weights=dyn.input_weights[state], cov=dyn.covs[state]),
        emissions=ParamsLGSSMEmissions(weights=emis.weights[0], bias=emis.biases[0],
                                       input_weights=emis.input_weights[0], cov=emis.covs[0]))


class TestSingleState:
    """With K = 1 the approximate posterior is the exact LGSSM posterior and the ELBO is log p(y)."""
    state_dim, emission_dim, input_dim, num_timesteps = 2, 4, 2, 40
    model = GaussianSLDS(1, state_dim, emission_dim, input_dim, emission_cov_type="full", damping=0.0,
                         **mle_kwargs(state_dim, emission_dim))
    params, props = model.initialize(jr.PRNGKey(0))
    inputs = jr.normal(jr.PRNGKey(1), (num_timesteps, input_dim))
    _, _, emissions = model.sample(params, jr.PRNGKey(2), num_timesteps, inputs)
    lgssm = LinearGaussianSSM(state_dim, emission_dim, input_dim, has_dynamics_bias=True, has_emissions_bias=True)
    reference = lgssm.smoother(to_lgssm(params), emissions, inputs)

    def test_posterior_matches_kalman_smoother(self):
        posterior, _ = self.model.approximate_posterior(self.params, self.emissions, self.inputs, num_iters=1)
        qx = jax.tree_util.tree_map(lambda a: a[0], posterior).continuous
        assert jnp.allclose(qx.smoothed_means, self.reference.smoothed_means, atol=TOL)
        assert jnp.allclose(qx.smoothed_covariances, self.reference.smoothed_covariances, atol=TOL)
        assert jnp.allclose(qx.smoothed_cross_covariances, self.reference.smoothed_cross_covariances, atol=TOL)

    def test_elbo_is_marginal_log_likelihood(self):
        _, elbos = self.model.approximate_posterior(self.params, self.emissions, self.inputs, num_iters=2)
        assert jnp.allclose(elbos[-1], self.reference.marginal_loglik, rtol=TOL)

    def test_em_step_matches_lgssm(self):
        """One maximum-likelihood M-step equals LinearGaussianSSM's on the dynamics and emissions (the SLDS keeps
        the initial distribution of x frozen; the LGSSM M-step cannot freeze it alone)."""
        fitted, lps = self.model.fit_em(self.params, self.props, self.emissions, self.inputs, num_iters=1, verbose=False)
        _, lg_props = self.lgssm.initialize(jr.PRNGKey(0))
        lg_fitted, lg_lps = self.lgssm.fit_em(to_lgssm(self.params), lg_props, self.emissions, self.inputs, num_iters=1, verbose=False)
        assert jnp.allclose(lps, lg_lps, rtol=TOL)
        ours, theirs = to_lgssm(fitted), lg_fitted
        for a, b in zip(jax.tree_util.tree_leaves((ours.dynamics, ours.emissions)), jax.tree_util.tree_leaves((theirs.dynamics, theirs.emissions))):
            assert jnp.allclose(a, b, atol=10 * TOL, rtol=10 * TOL)

    def test_em_trace_matches_lgssm(self):
        """With the initial distribution of x trainable too, five EM iterations reproduce LinearGaussianSSM's log-probability trace."""
        props = self.props._replace(dynamics=self.props.dynamics._replace(
            initial_mean=ParameterProperties(), initial_cov=ParameterProperties(constrainer=self.props.dynamics.initial_cov.constrainer)))
        _, lps = self.model.fit_em(self.params, props, self.emissions, self.inputs, num_iters=5, verbose=False)
        _, lg_props = self.lgssm.initialize(jr.PRNGKey(0))
        _, lg_lps = self.lgssm.fit_em(to_lgssm(self.params), lg_props, self.emissions, self.inputs, num_iters=5, verbose=False)
        assert jnp.allclose(lps, lg_lps, rtol=10 * TOL)

    def test_stiff_dynamics_in_float32(self):
        """Nearly deterministic dynamics (Q = 1e-4, T = 500): the information-form smoother stays within 1% of the Kalman smoother."""
        D, N, T = 2, 4, 500
        model = GaussianSLDS(1, D, N, emission_cov_type="full", damping=0.0)
        params, _ = model.initialize(jr.PRNGKey(0))
        params = params._replace(dynamics=params.dynamics._replace(covs=1e-4 * jnp.tile(jnp.eye(D), (1, 1, 1)),
                                                                   weights=0.999 * jnp.tile(jnp.eye(D), (1, 1, 1))))
        _, _, emissions = model.sample(params, jr.PRNGKey(3), T)
        posterior, _ = model.approximate_posterior(params, emissions, num_iters=1)
        reference = LinearGaussianSSM(D, N, has_dynamics_bias=True, has_emissions_bias=True).smoother(to_lgssm(params), emissions)
        means = posterior.continuous.smoothed_means[0]
        assert jnp.all(jnp.isfinite(means))
        assert jnp.abs(means - reference.smoothed_means).max() < 1e-2 * jnp.abs(reference.smoothed_means).max()


class TestTwoStates:
    """A K = 2 model small enough to enumerate the discrete sequences."""
    num_states, state_dim, emission_dim, num_timesteps = 2, 2, 3, 5
    model = GaussianSLDS(num_states, state_dim, emission_dim, emission_cov_type="full", damping=0.0)
    params, props = model.initialize(jr.PRNGKey(3))
    params = params._replace(transitions=params.transitions._replace(transition_matrix=jnp.array([[0.9, 0.1], [0.2, 0.8]])))
    _, _, emissions = model.sample(params, jr.PRNGKey(4), num_timesteps)
    inputs = jnp.zeros((num_timesteps, 0))

    def _log_marginal_likelihood(self):
        """log p(y) by summing the LGSSM likelihood of every discrete sequence (time-varying dynamics)."""
        log_P, log_pi = jnp.log(self.params.transitions.transition_matrix), jnp.log(self.params.initial.probs)

        def _one(z):
            z = jnp.array(z)
            log_pz = log_pi[z[0]] + log_P[z[:-1], z[1:]].sum()
            return log_pz + lgssm_filter(to_lgssm(self.params, z), self.emissions, self.inputs).marginal_loglik

        return logsumexp(jnp.array([_one(z) for z in itertools.product(range(self.num_states), repeat=self.num_timesteps)]))

    def test_elbo_is_a_lower_bound(self):
        _, elbos = self.model.approximate_posterior(self.params, self.emissions, num_iters=20)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] <= self._log_marginal_likelihood() + 1e-3
        assert jnp.all(jnp.diff(elbos) >= -1e-3 * jnp.abs(elbos[:-1]))


class TestNewton:
    def test_line_search_backtracks(self):
        """A direction that overshoots is shortened; NaN objective values are treated as no decrease."""
        f = lambda x: jnp.where(jnp.all(jnp.abs(x) < 10.0), jnp.sum(x ** 2), jnp.nan)
        x = jnp.ones((6, 2))
        dx = -20.0 * x   # the full step lands at -19 x, where f is NaN
        g = jax.grad(lambda x: jnp.sum(x ** 2))(x)
        step = backtracking_line_search(f, x, dx, g)
        assert 0.0 < step < 1.0
        assert f(x + step * dx) <= f(x) + 0.2 * step * jnp.sum(g * dx)


class TestLogSpaceChain:
    def test_smoother_survives_underflowing_messages(self):
        """A left-to-right chain 0 -> 1 -> 2 whose data favor state 0 and then state 2, so the posterior passes
        through state 1 for one step. In float32 ``hmm_filter`` underflows state 1 to zero and fails at the
        switch; the log-space smoother matches exact enumeration."""
        T, K, gap = 40, 3, 150.0
        pi0 = jnp.array([1.0, 0.0, 0.0])
        P = jnp.array([[1 - 1e-3, 1e-3, 0.0], [0.0, 1 - 1e-3, 1e-3], [0.0, 0.0, 1.0]])
        ll = gap * jax.nn.one_hot(jnp.where(jnp.arange(T) < 20, 0, 2), K)
        # every monotone path: 0 before t1, 1 on [t1, t2), 2 from t2 on, with 1 <= t1 <= t2 <= T
        t1, t2 = jnp.array([(a, b) for a in range(1, T + 1) for b in range(a, T + 1)]).T
        paths = jnp.where(jnp.arange(T) < t1[:, None], 0, jnp.where(jnp.arange(T) < t2[:, None], 1, 2))
        log_joint = jnp.log(P)[paths[:, :-1], paths[:, 1:]].sum(1) + ll[jnp.arange(T), paths].sum(1)
        exact = jax.nn.softmax(log_joint)
        exact_marginals = jnp.stack([(exact[:, None] * (paths == k)).sum(0) for k in range(K)], 1)
        _, smoothed, _ = log_space_hmm_smoother(pi0, jnp.log(P), ll)
        assert jnp.allclose(smoothed, exact_marginals, atol=1e-3)


def test_exactness_under_x64():
    """Re-run this file with 64-bit floats, where the tolerances tighten to 1e-7."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-3000:]
test_exactness_under_x64.__test__ = not X64     # already under x64: nothing to re-run
