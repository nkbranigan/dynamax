"""Tests for the SLDS components on their own."""
import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from dynamax.linear_gaussian_ssm.block_tridiag import BlockTridiagPotentials, block_tridiag_mvn_sample, block_tridiag_mvn_smoother
from dynamax.slds import (
    LinearGaussianSLDSDynamics, LinearGaussianSLDSEmissions, SLDSInitialState, StationarySLDSTransitions)

TOL = 2e-3


def random_gaussian_posterior(key, num_timesteps, state_dim):
    """Make a random Gaussian posterior over the continuous states."""
    k1, k2, k3 = jr.split(key, 3)
    T, D = num_timesteps, state_dim
    A = 0.3 * jr.normal(k1, (T, D, D))
    J_diag = 3.0 * jnp.eye(D) + A @ jnp.swapaxes(A, 1, 2)
    J_lower = 0.3 * jr.normal(k2, (T - 1, D, D))
    h = jr.normal(k3, (T, D))
    return block_tridiag_mvn_smoother(BlockTridiagPotentials(J_diag, J_lower, h))


class TestExpectedLogLikelihoods:
    num_states, state_dim, emission_dim, input_dim, num_timesteps = 3, 2, 5, 0, 20

    def _case(self, single_subspace):
        K, D, N, M, T = self.num_states, self.state_dim, self.emission_dim, self.input_dim, self.num_timesteps
        dyn = LinearGaussianSLDSDynamics(K, D, M, cov_type="full")
        emis = LinearGaussianSLDSEmissions(K, D, N, M, single_subspace=single_subspace, cov_type="full")
        dyn_params, _ = dyn.initialize(jr.PRNGKey(10))
        emis_params, _ = emis.initialize(jr.PRNGKey(11))
        inputs = jr.normal(jr.PRNGKey(12), (T, M))
        emissions = jr.normal(jr.PRNGKey(13), (T, N))
        return dyn, emis, dyn_params, emis_params, inputs, emissions

    @pytest.mark.parametrize("single_subspace", [True, False])
    def test_expected_log_likelihoods(self, single_subspace):
        """With zero covariance the expected log likelihoods equal the log likelihoods at the mean; otherwise
        they match averages over samples."""
        dyn, emis, dyn_params, emis_params, inputs, emissions = self._case(single_subspace)
        T, D = self.num_timesteps, self.state_dim
        qx = random_gaussian_posterior(jr.PRNGKey(16), T, D)
        means = qx.smoothed_means
        # zero covariance
        point = qx._replace(smoothed_covariances=jnp.zeros((T, D, D)),
                            smoothed_cross_covariances=jnp.einsum("ti,tj->tij", means[:-1], means[1:]))
        assert jnp.allclose(dyn.expected_log_likelihoods(dyn_params, point, inputs),
                            dyn.log_likelihoods(dyn_params, means, inputs), atol=TOL, rtol=TOL)
        assert jnp.allclose(emis.expected_log_likelihoods(emis_params, emissions, point, inputs),
                            emis.log_likelihoods(emis_params, emissions, means, inputs), atol=TOL, rtol=TOL)
        # samples from the posterior with a fixed key
        num_samples = 4000
        xs = vmap(lambda k: block_tridiag_mvn_sample(k, qx))(jr.split(jr.PRNGKey(17), num_samples))
        for exact, per_sample in [
            (dyn.expected_log_likelihoods(dyn_params, qx, inputs), vmap(lambda x: dyn.log_likelihoods(dyn_params, x, inputs))(xs)),
            (emis.expected_log_likelihoods(emis_params, emissions, qx, inputs), vmap(lambda x: emis.log_likelihoods(emis_params, emissions, x, inputs))(xs))]:
            totals = per_sample.sum(axis=(1, 2))
            stderr = totals.std() / jnp.sqrt(num_samples)
            assert jnp.abs(exact.sum() - totals.mean()) < 4 * stderr + 1e-3 * jnp.abs(exact.sum())
            assert jnp.allclose(exact, per_sample.mean(0), atol=20 * per_sample.std(0).max() / jnp.sqrt(num_samples) + 1e-3)


class TestDiscreteComponents:
    def test_large_counts_keep_a_positive_floor(self):
        """The floor grows with the row total, so it does not round to zero in float32."""
        trans = StationarySLDSTransitions(2)
        params, props = trans.initialize(jr.PRNGKey(0))
        new, _ = trans.m_step(params, props, jnp.array([[[3e6, 0.0], [10.0, 20.0]]]), None)
        assert jnp.all(new.transition_matrix > 0) and jnp.allclose(new.transition_matrix.sum(1), 1.0)

    def test_small_concentrations_clip_the_pseudo_counts(self):
        """Concentrations below one subtract from the counts; zero counts clip at zero instead of going negative."""
        trans = StationarySLDSTransitions(3, concentration=0.5)
        params, props = trans.initialize(jr.PRNGKey(0))
        new, _ = trans.m_step(params, props, jnp.array([[[5.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]]), None)
        assert jnp.all(new.transition_matrix >= 0) and jnp.allclose(new.transition_matrix.sum(1), 1.0)
        assert jnp.isfinite(trans.log_prior(new))
        initial = SLDSInitialState(3, initial_probs_concentration=0.5)
        params, props = initial.initialize(jr.PRNGKey(0))
        new, _ = initial.m_step(params, props, jnp.array([[4.0, 0.0, 0.0]]), None)
        assert jnp.all(new.probs >= 0) and jnp.allclose(new.probs.sum(), 1.0)


class TestValidationAndInversion:
    def test_wrong_covariance_shape_is_rejected(self):
        with pytest.raises(ValueError):
            LinearGaussianSLDSDynamics(2, 2).initialize(jr.PRNGKey(0), covs=0.1 * jnp.ones((2, 2)))
        with pytest.raises(ValueError):
            LinearGaussianSLDSEmissions(2, 2, 4).initialize(jr.PRNGKey(0), covs=jnp.tile(jnp.eye(4), (1, 1, 1)))

    def test_invert_recovers_the_latents(self):
        emis = LinearGaussianSLDSEmissions(2, 2, 5, input_dim=1)
        params, _ = emis.initialize(jr.PRNGKey(0), biases=jr.normal(jr.PRNGKey(3), (1, 5)),
                                    input_weights=jr.normal(jr.PRNGKey(4), (1, 5, 1)))
        x = jr.normal(jr.PRNGKey(1), (30, 2))
        inputs = jr.normal(jr.PRNGKey(2), (30, 1))
        y = x @ params.weights[0].T + inputs @ params.input_weights[0].T + params.biases[0]
        assert jnp.allclose(emis.invert(params, y, inputs), x, atol=TOL)
