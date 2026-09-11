"""Tests for Gaussian HMM cross-validation over candidate state counts."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import jit, vmap

from dynamax.hidden_markov_model import DiagonalGaussianHMM, GaussianHMM, cross_validate_num_states
from dynamax.hidden_markov_model.cross_validation import (
    _cross_validate_from_params,
    _leave_one_out_folds,
    _prepare_models,
    _project_markov_params,
)


NUM_BATCHES = 3
NUM_TIMESTEPS = 24
EMISSION_DIM = 2
NUM_ITERS = 3


def _make_emissions():
    """Create a small deterministic collection of two-dimensional sequences."""
    time = jnp.linspace(0.0, 2.0 * jnp.pi, NUM_TIMESTEPS)
    return jnp.stack(
        [jnp.column_stack((jnp.cos(time + shift), jnp.sin(2.0 * time + shift))) for shift in (0.0, 0.4, 0.8)]
    )


def _make_models():
    return [
        GaussianHMM(
            num_states,
            EMISSION_DIM,
            transition_matrix_stickiness=3.0,
            emission_prior_concentration=0.2,
            emission_prior_scale=0.3,
            emission_prior_extra_df=0.5,
        )
        for num_states in (2, 3)
    ]


def _make_explicit_initial_params(models):
    means = jnp.array([[-0.8, 0.0], [0.2, 0.7], [0.9, -0.4]])
    initialized = []
    for model in models:
        num_states = model.num_states
        initial_probs = jnp.arange(1, num_states + 1, dtype=jnp.float32)
        initial_probs /= initial_probs.sum()
        transition_matrix = 0.8 * jnp.eye(num_states) + 0.2 / num_states
        covs = jnp.tile(0.5 * jnp.eye(EMISSION_DIM), (num_states, 1, 1))
        initialized.append(
            model.initialize(
                initial_probs=initial_probs,
                transition_matrix=transition_matrix,
                emission_means=means[:num_states],
                emission_covariances=covs,
            )
        )
    return initialized


def test_cross_validate_num_states_returns_finite_scores():
    emissions = _make_emissions()
    avg_val_lls, val_lls = cross_validate_num_states(
        _make_models(), emissions, jr.PRNGKey(0), num_iters=NUM_ITERS
    )

    assert avg_val_lls.shape == (2,)
    assert val_lls.shape == (2, NUM_BATCHES)
    assert jnp.all(jnp.isfinite(val_lls))
    assert jnp.allclose(avg_val_lls, val_lls.mean(axis=1))


def test_prepare_models_initializes_each_model_on_full_training_set(monkeypatch):
    emissions = _make_emissions()
    models = _make_models()
    initialization_datasets = []
    original_initialize = GaussianHMM.initialize

    def recording_initialize(self, *args, **kwargs):
        initialization_datasets.append(np.asarray(kwargs["emissions"]))
        return original_initialize(self, *args, **kwargs)

    monkeypatch.setattr(GaussianHMM, "initialize", recording_initialize)
    setup = _prepare_models(models, emissions, key=jr.PRNGKey(0))

    assert len(initialization_datasets) == len(models)
    for initialization_dataset in initialization_datasets:
        np.testing.assert_array_equal(initialization_dataset, np.asarray(emissions))
    assert setup.params.emissions.means.shape == (
        len(models),
        max(model.num_states for model in models),
        EMISSION_DIM,
    )


def test_padded_cross_validation_matches_unpadded_fits_from_same_start():
    emissions = _make_emissions()
    models = _make_models()
    initialized = _make_explicit_initial_params(models)
    folds = _leave_one_out_folds(emissions)

    setup = _prepare_models(models, emissions, initial_params=initialized)
    padded_val_lls = _cross_validate_from_params(setup, emissions, num_iters=NUM_ITERS)

    reference_rows = []
    for model, (params, props) in zip(models, initialized):
        def _fit_fold(train_emissions, val_emissions):
            fit_params, _ = model.fit_em(
                params,
                props,
                train_emissions,
                num_iters=NUM_ITERS,
                verbose=False,
            )
            return model.marginal_log_prob(fit_params, val_emissions)

        reference_rows.append(jit(vmap(_fit_fold))(folds, emissions))

    reference_val_lls = jnp.stack(reference_rows)
    assert jnp.allclose(padded_val_lls, reference_val_lls, rtol=2e-4, atol=2e-3)


def test_project_markov_params_makes_padded_states_inert():
    initial_probs = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    transition_matrix = jnp.tile(
        jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4),
        (2, 1, 1),
    )
    state_mask = jnp.array([True, True, False, False])

    projected_initial, projected_transitions = _project_markov_params(
        initial_probs, transition_matrix, state_mask
    )

    assert jnp.allclose(
        projected_initial,
        jnp.array(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0],
                [4.0 / 7.0, 3.0 / 7.0, 0.0, 0.0],
            ]
        ),
    )
    assert jnp.allclose(projected_transitions[:, :2, 2:], 0.0)
    assert jnp.allclose(projected_transitions[:, 2:], jnp.broadcast_to(jnp.eye(4)[2:], (2, 2, 4)))
    assert jnp.allclose(projected_transitions.sum(axis=-1), 1.0)


def test_cross_validate_num_states_validates_inputs():
    emissions = _make_emissions()
    key = jr.PRNGKey(0)

    with pytest.raises(ValueError, match="at least one"):
        cross_validate_num_states([], emissions, key, num_iters=1)

    with pytest.raises(TypeError, match="ordinary GaussianHMM"):
        cross_validate_num_states([DiagonalGaussianHMM(2, EMISSION_DIM)], emissions, key, num_iters=1)

    with pytest.raises(ValueError, match="at least two"):
        cross_validate_num_states(_make_models(), emissions[:1], key, num_iters=1)

    with pytest.raises(ValueError, match="same emission dimension"):
        cross_validate_num_states([GaussianHMM(2, 3)], emissions, key, num_iters=1)

    with pytest.raises(ValueError, match="nonnegative"):
        cross_validate_num_states(_make_models(), emissions, key, num_iters=-1)
