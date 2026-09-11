"""Tests for standard HMM initial-state and transition initialization."""

import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions


def test_initial_state_initialize_uses_jax_dirichlet():
    """The initial-state component samples with the same key and dtype as JAX."""
    component = StandardHMMInitialState(num_states=4)
    key = jr.PRNGKey(0)

    params, _ = component.initialize(key)
    expected = jr.dirichlet(
        jr.split(key)[0],
        component.initial_probs_concentration,
        dtype=component.initial_probs_concentration.dtype,
    )

    assert jnp.array_equal(params.probs, expected)


def test_transitions_initialize_uses_jax_dirichlet():
    """The transition component samples each row with JAX's Dirichlet sampler."""
    component = StandardHMMTransitions(num_states=4, stickiness=2.0)
    key = jr.PRNGKey(0)

    params, _ = component.initialize(key)
    expected = jr.dirichlet(
        key,
        component.concentration,
        dtype=component.concentration.dtype,
    )

    assert jnp.array_equal(params.transition_matrix, expected)


def test_standard_initialization_is_jittable_and_vmappable():
    """Both shared initialization paths compose with an outer JIT and vmap."""
    num_states = 4
    num_models = 3
    keys = jr.split(jr.PRNGKey(1), num_models)
    initial_component = StandardHMMInitialState(num_states)
    transition_component = StandardHMMTransitions(num_states)

    initialize_initial = jax.jit(jax.vmap(lambda key: initial_component.initialize(key)[0].probs))
    initialize_transitions = jax.jit(
        jax.vmap(lambda key: transition_component.initialize(key)[0].transition_matrix)
    )

    initial_probs = initialize_initial(keys)
    transition_matrices = initialize_transitions(keys)

    assert initial_probs.shape == (num_models, num_states)
    assert transition_matrices.shape == (num_models, num_states, num_states)
    assert initial_probs.dtype == initial_component.initial_probs_concentration.dtype
    assert transition_matrices.dtype == transition_component.concentration.dtype
    assert jnp.all(jnp.isfinite(initial_probs))
    assert jnp.all(jnp.isfinite(transition_matrices))
    assert jnp.all(initial_probs >= 0)
    assert jnp.all(transition_matrices >= 0)
    assert jnp.allclose(initial_probs.sum(axis=-1), 1.0)
    assert jnp.allclose(transition_matrices.sum(axis=-1), 1.0)


def test_standard_initialization_preserves_concentration_dtype():
    """JAX sampling preserves the dtype used by the Dirichlet concentration."""
    initial_component = StandardHMMInitialState(3)
    transition_component = StandardHMMTransitions(3)
    initial_component.initial_probs_concentration = jnp.full((3,), 1.1, dtype=jnp.float16)
    transition_component.concentration = jnp.full((3, 3), 1.1, dtype=jnp.float16)

    initial_params, _ = initial_component.initialize(jr.PRNGKey(2))
    transition_params, _ = transition_component.initialize(jr.PRNGKey(3))

    assert initial_params.probs.dtype == jnp.float16
    assert transition_params.transition_matrix.dtype == jnp.float16


def test_standard_initialization_preserves_explicit_parameters():
    """Explicit parameters bypass random initialization and do not require a key."""
    initial_probs = jnp.array([0.2, 0.3, 0.5])
    transition_matrix = jnp.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    initial_params, _ = StandardHMMInitialState(3).initialize(initial_probs=initial_probs)
    transition_params, _ = StandardHMMTransitions(3).initialize(transition_matrix=transition_matrix)

    assert jnp.array_equal(initial_params.probs, initial_probs)
    assert jnp.array_equal(transition_params.transition_matrix, transition_matrix)
