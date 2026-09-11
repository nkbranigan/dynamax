"""Tests for cross_validate_num_states."""
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import jit, tree_util, vmap

import dynamax.hidden_markov_model as models
from dynamax.hidden_markov_model.cross_validation import (
    _leave_one_out_folds, _prepare, _sample_padded_inits, cross_validate_num_states)

NUM_BATCHES = 3
NUM_TIMESTEPS = 40
NUM_ITERS = 10
ALL_NUM_STATES = [2, 3, 4]

# (model class, constructor kwargs, emission init method, needs inputs)
CONFIGS = [
    (models.GaussianHMM, dict(emission_dim=2, transition_matrix_stickiness=10.), "kmeans", False),
    (models.DiagonalGaussianHMM, dict(emission_dim=2), "prior", False),
    (models.PoissonHMM, dict(emission_dim=2), "prior", False),
    (models.CategoricalHMM, dict(emission_dim=2, num_classes=3), "prior", False),
    # No emission prior: padded states produce NaN parameters that masking must isolate.
    (models.LinearRegressionHMM, dict(emission_dim=2, input_dim=2), "prior", True),
]


def _make_data(cls, kwargs, has_inputs, key):
    """Sample batched emissions (and inputs) from a 4-state model of the given class."""
    true_model = cls(num_states=4, **kwargs)
    key_params, key_sample, key_inputs = jr.split(key, 3)
    params, _ = true_model.initialize(key_params)
    inputs = None
    if has_inputs:
        inputs = jr.normal(key_inputs, (NUM_BATCHES, NUM_TIMESTEPS, kwargs["input_dim"]))

    def sample(key, u):
        return true_model.sample(params, key, num_timesteps=NUM_TIMESTEPS, inputs=u)[1]

    keys = jr.split(key_sample, NUM_BATCHES)
    emissions = vmap(sample)(keys, inputs) if has_inputs else vmap(lambda k: sample(k, None))(keys)
    return emissions, inputs


def _per_model_val_lls(setup, key_draw, emissions, inputs, num_iters):
    """Reference: fit each unpadded model separately from the same initialization."""
    init_probs, trans_mats = _sample_padded_inits(key_draw, setup.draw_conc, setup.masks)
    train_y, train_u = _leave_one_out_folds(emissions), _leave_one_out_folds(inputs)
    rows = []
    for i, model in enumerate(setup.models):
        k = model.num_states
        params_i = tree_util.tree_map(lambda x: x[i], setup.params0)
        params = params_i._replace(
            initial=params_i.initial._replace(probs=init_probs[i, :k]),
            transitions=params_i.transitions._replace(transition_matrix=trans_mats[i, :k, :k]),
            emissions=setup.unpadded_emissions[i])

        def _fit_fold(y_train, y_val, u_train, u_val):
            fit_params, _ = model.fit_em(params, setup.props, y_train, inputs=u_train,
                                         num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val, inputs=u_val)

        in_axes = (0, 0, None if train_u is None else 0, None if inputs is None else 0)
        rows.append(jit(vmap(_fit_fold, in_axes=in_axes))(train_y, emissions, train_u, inputs))
    return jnp.stack(rows)


@pytest.mark.parametrize(["cls", "kwargs", "init_method", "has_inputs"], CONFIGS)
def test_padded_cv_matches_per_model_fits(cls, kwargs, init_method, has_inputs):
    """The single-compile padded sweep must reproduce one-model-per-k fits."""
    key_data, key_cv = jr.split(jr.PRNGKey(0))
    emissions, inputs = _make_data(cls, kwargs, has_inputs, key_data)

    avg_val_lls, val_lls = cross_validate_num_states(
        cls, ALL_NUM_STATES, emissions, key_cv, inputs=inputs, num_iters=NUM_ITERS,
        init_method=init_method, model_kwargs=kwargs)
    assert val_lls.shape == (len(ALL_NUM_STATES), NUM_BATCHES)
    assert jnp.all(jnp.isfinite(val_lls))

    # Rebuild the same initialization (mirrors the key split inside cross_validate_num_states).
    key_init, key_draw = jr.split(key_cv)
    setup = _prepare(cls, ALL_NUM_STATES, emissions, key_init, init_method, kwargs, {})
    reference = _per_model_val_lls(setup, key_draw, emissions, inputs, NUM_ITERS)

    assert jnp.allclose(val_lls, reference, rtol=2e-3, atol=5e-2), (val_lls, reference)
    assert jnp.allclose(avg_val_lls, reference.mean(axis=1), rtol=2e-3, atol=5e-2)


def test_padded_states_stay_inert():
    """Padded initial/transition parameters carry no probability mass."""
    from dynamax.hidden_markov_model.cross_validation import _mask_params
    model = models.GaussianHMM(num_states=5, emission_dim=2)
    params, _ = model.initialize(jr.PRNGKey(0))
    mask = jnp.array([True, True, True, False, False])
    masked = _mask_params(params, mask)
    assert jnp.allclose(masked.initial.probs[3:], 0.0)
    assert jnp.allclose(masked.initial.probs.sum(), 1.0)
    A = masked.transitions.transition_matrix
    assert jnp.allclose(A[:3, 3:], 0.0)            # live states never enter padded ones
    assert jnp.allclose(A[3:], jnp.eye(5)[3:])     # padded rows are self-loops
    assert jnp.allclose(A.sum(axis=1), 1.0)


def test_repeat_call_is_deterministic_and_cached():
    key_data, key_cv = jr.split(jr.PRNGKey(1))
    emissions, _ = _make_data(models.GaussianHMM, dict(emission_dim=2), False, key_data)
    kwargs = dict(emission_dim=2)
    a = cross_validate_num_states(models.GaussianHMM, ALL_NUM_STATES, emissions, key_cv,
                                  num_iters=NUM_ITERS, init_method="kmeans", model_kwargs=kwargs)
    b = cross_validate_num_states(models.GaussianHMM, ALL_NUM_STATES, emissions, key_cv,
                                  num_iters=NUM_ITERS, init_method="kmeans", model_kwargs=kwargs)
    assert jnp.array_equal(a[1], b[1])


def test_requires_batched_emissions():
    single_sequence = jnp.zeros((NUM_TIMESTEPS, 2))
    with pytest.raises(ValueError):
        cross_validate_num_states(models.GaussianHMM, ALL_NUM_STATES, single_sequence, jr.PRNGKey(0),
                                  model_kwargs=dict(emission_dim=2))


def _initial_params_comparison(num_iters):
    """Padded sweep vs. one-model-per-k fits, both from the same initial parameters."""
    key_data, key_init = jr.split(jr.PRNGKey(2))
    kwargs = dict(emission_dim=2, transition_matrix_stickiness=10.)
    emissions, _ = _make_data(models.GaussianHMM, kwargs, False, key_data)
    candidates = [models.GaussianHMM(num_states=k, **kwargs) for k in ALL_NUM_STATES]
    inits = [m.initialize(key=key, method="kmeans", emissions=emissions)
             for m, key in zip(candidates, jr.split(key_init, len(candidates)))]

    _, val_lls = cross_validate_num_states(models.GaussianHMM, ALL_NUM_STATES, emissions, jr.PRNGKey(0),
                                           num_iters=num_iters, model_kwargs=kwargs, initial_params=inits)

    train_y = _leave_one_out_folds(emissions)
    reference = []
    for model, (params, props) in zip(candidates, inits):
        def _fit_fold(y_train, y_val):
            fit_params, _ = model.fit_em(params, props, y_train, num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val)
        reference.append(jit(vmap(_fit_fold))(train_y, emissions))
    return val_lls, jnp.stack(reference)


def test_initial_params_reproduces_per_model_fits():
    """Same starting parameters => same fits, up to float32 rounding amplified by EM."""
    val_lls, reference = _initial_params_comparison(NUM_ITERS)
    assert jnp.allclose(val_lls, reference, rtol=1e-2, atol=0.5), (val_lls, reference)


def test_initial_params_reproduces_per_model_fits_float64():
    """In float64 the padded sweep is exact: the only approximation is rounding."""
    import jax
    jax.config.update("jax_enable_x64", True)
    try:
        val_lls, reference = _initial_params_comparison(NUM_ITERS)
        assert val_lls.dtype == jnp.float64
        assert jnp.allclose(val_lls, reference, rtol=1e-7, atol=1e-6), (val_lls, reference)
    finally:
        jax.config.update("jax_enable_x64", False)
