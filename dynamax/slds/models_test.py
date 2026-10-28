"""Tests for ``GaussianSLDS``."""
import os
import subprocess
import sys

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from dynamax.parameters import ParameterProperties
from dynamax.slds import GaussianSLDS, ParamsSLDS
from dynamax.utils.utils import monotonically_increasing

X64 = jax.config.read("jax_enable_x64")


def well_separated_model():
    """A K = 2 model whose states rotate in opposite directions around different points, so z is identifiable from y."""
    model = GaussianSLDS(2, 2, 4)
    params, props = model.initialize(jr.PRNGKey(0))
    theta = 0.5
    rot = lambda s: 0.98 * jnp.array([[jnp.cos(s), -jnp.sin(s)], [jnp.sin(s), jnp.cos(s)]])
    params = params._replace(
        initial=params.initial._replace(probs=jnp.array([0.5, 0.5])),
        transitions=params.transitions._replace(transition_matrix=jnp.array([[0.95, 0.05], [0.05, 0.95]])),
        dynamics=params.dynamics._replace(weights=jnp.stack([rot(theta), rot(-theta)]),
                                          biases=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
                                          covs=jnp.tile(0.05 * jnp.eye(2), (2, 1, 1))),
        emissions=params.emissions._replace(covs=0.1 * jnp.ones_like(params.emissions.covs)))
    return model, params, props


class TestPosterior:
    model, params, props = well_separated_model()
    z, x, y = model.sample(params, jr.PRNGKey(5), 200)

    def test_recovers_states_with_true_parameters(self):
        posterior, elbos = self.model.approximate_posterior(self.params, self.y, num_iters=10)
        posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)
        z_hat = self.model.most_likely_states(self.params, self.y, posterior)
        accuracy = jnp.mean(z_hat == self.z)
        assert accuracy > 0.85
        assert jnp.mean(posterior.discrete.smoothed_probs.argmax(1) == self.z) > 0.85
        assert jnp.mean((self.model.expected_emissions(self.params, posterior) - self.y) ** 2) < jnp.mean((self.y - self.y.mean(0)) ** 2)


def _sample_batch(key, num_sequences, num_timesteps):
    model, params, _ = well_separated_model()
    return vmap(lambda k: model.sample(params, k, num_timesteps)[2])(jr.split(key, num_sequences))


class TestFitEM:
    model, true_params, _ = well_separated_model()
    y = model.sample(true_params, jr.PRNGKey(6), 150)[2]
    y_batch = _sample_batch(jr.PRNGKey(7), 3, 60)

    def test_objective_increases_without_damping(self):
        model = GaussianSLDS(2, 2, 4, damping=0.0, perturb_unused_states=False)
        params, props = model.initialize(jr.PRNGKey(0))
        _, lps = model.fit_em(params, props, self.y, num_iters=15, verbose=False)
        assert jnp.all(jnp.isfinite(lps))
        assert monotonically_increasing(lps, rtol=0 if X64 else 1e-5)

    def test_fit_with_damping_and_batches(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        fitted, lps, posterior = model.fit_em(params, props, self.y_batch, num_iters=5, verbose=False, return_posterior=True)
        assert lps.shape == (5,) and jnp.all(jnp.isfinite(lps))
        assert lps[-1] > lps[0]
        assert posterior.discrete.smoothed_probs.shape == (3, 60, 2)
        assert posterior.continuous.smoothed_means.shape == (3, 60, 2)
        assert jnp.allclose(fitted.transitions.transition_matrix.sum(1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("dynamics_cov_type, emission_cov_type, single_subspace",
                             [("full", "diag", True), ("diag", "full", False), ("diag", "diag", False)])
    def test_variants_with_inputs(self, dynamics_cov_type, emission_cov_type, single_subspace):
        model = GaussianSLDS(2, 2, 4, input_dim=2, single_subspace=single_subspace,
                             dynamics_cov_type=dynamics_cov_type, emission_cov_type=emission_cov_type)
        params, props = model.initialize(jr.PRNGKey(1))
        inputs = jr.normal(jr.PRNGKey(2), (80, 2))
        _, _, y = model.sample(params, jr.PRNGKey(3), 80, inputs)
        params, props = model.initialize(jr.PRNGKey(4))
        fitted, lps = model.fit_em(params, props, y, inputs, num_iters=4, verbose=False)
        assert jnp.all(jnp.isfinite(lps)) and lps[-1] > lps[0]
        assert fitted.emissions.weights.shape == (1 if single_subspace else 2, 4, 2)
        assert fitted.dynamics.covs.shape == ((2, 2) if dynamics_cov_type == "diag" else (2, 2, 2))
        assert fitted.emissions.covs.shape[1:] == ((4,) if emission_cov_type == "diag" else (4, 4))
        if emission_cov_type == "diag":
            assert jnp.all(fitted.emissions.covs > 0)

    def test_verbose_matches_silent(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        _, lps_silent = model.fit_em(params, props, self.y, num_iters=4, verbose=False)
        _, lps_verbose = model.fit_em(params, props, self.y, num_iters=4, verbose=True, print_every=2)
        assert jnp.allclose(lps_silent, lps_verbose, rtol=1e-5)

    def test_damping_keeps_exact_zeros_finite(self):
        """Zero probabilities (left-to-right transitions, a one-hot initial distribution) survive damping."""
        model = GaussianSLDS(3, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        y = model.sample(params, jr.PRNGKey(1), 60)[2]
        params = params._replace(initial=params.initial._replace(probs=jnp.array([1.0, 0.0, 0.0])),
                                 transitions=params.transitions._replace(transition_matrix=jnp.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]])))
        fitted, lps = model.fit_em(params, props, y, num_iters=3, verbose=False)
        assert jnp.all(jnp.isfinite(lps))
        assert jnp.array_equal(fitted.initial.probs, jnp.array([1.0, 0.0, 0.0]))
        assert jnp.all(fitted.transitions.transition_matrix[jnp.tril_indices(3, -1)] == 0.0)
        assert jnp.allclose(fitted.transitions.transition_matrix.sum(1), 1.0)

    def test_damping_stays_near_the_old_parameters(self):
        """With damping 0.75 the transition matrix stays closer to the old one than to the undamped update, and the
        dynamics are not damped."""
        fits = {}
        for damping in (0.0, 0.75):
            model = GaussianSLDS(2, 2, 4, damping=damping)
            params, props = model.initialize(jr.PRNGKey(0))
            fits[damping], _ = model.fit_em(params, props, self.y, num_iters=1, verbose=False)
        old, new, damped = (p.transitions.transition_matrix for p in (params, fits[0.0], fits[0.75]))
        assert jnp.abs(damped - old).max() < jnp.abs(damped - new).max()
        assert jnp.allclose(fits[0.75].dynamics.weights, fits[0.0].dynamics.weights, atol=1e-5)


class TestFreezing:
    model, true_params, _ = well_separated_model()
    y = model.sample(true_params, jr.PRNGKey(8), 60)[2]

    @staticmethod
    def freeze(props):
        return jax.tree_util.tree_map(lambda p: ParameterProperties(trainable=False, constrainer=p.constrainer), props,
                                      is_leaf=lambda n: isinstance(n, ParameterProperties))

    def test_frozen_components_do_not_move(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        props = ParamsSLDS(initial=self.freeze(props.initial), transitions=self.freeze(props.transitions),
                           dynamics=props.dynamics, emissions=self.freeze(props.emissions))
        fitted, lps = model.fit_em(params, props, self.y, num_iters=3, verbose=False)
        assert jnp.array_equal(fitted.initial.probs, params.initial.probs)
        assert jnp.array_equal(fitted.transitions.transition_matrix, params.transitions.transition_matrix)
        assert all(jnp.array_equal(a, b) for a, b in zip(fitted.emissions, params.emissions))
        assert not jnp.array_equal(fitted.dynamics.weights, params.dynamics.weights)
        # the initial distribution of x is frozen by default
        assert jnp.array_equal(fitted.dynamics.initial_mean, params.dynamics.initial_mean)

    def test_partial_freezing_raises(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        props = props._replace(dynamics=props.dynamics._replace(weights=ParameterProperties(trainable=False)))
        with pytest.raises(NotImplementedError):
            model.fit_em(params, props, self.y, num_iters=1, verbose=False)
        params, props = model.initialize(jr.PRNGKey(0))
        props = props._replace(emissions=props.emissions._replace(covs=ParameterProperties(trainable=False, constrainer=props.emissions.covs.constrainer)))
        with pytest.raises(NotImplementedError):
            model.fit_em(params, props, self.y, num_iters=1, verbose=False)


def test_monotonicity_under_x64():
    """Re-run the monotonicity test of this file with 64-bit floats, where it is strict."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "increases"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-3000:]
test_monotonicity_under_x64.__test__ = not X64     # already under x64: nothing to re-run
