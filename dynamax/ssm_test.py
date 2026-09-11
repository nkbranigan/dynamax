"""Tests for the shared state-space model fitting methods."""

from functools import partial

import jax.numpy as jnp
import jax.random as jr
import pytest

from jax import jit, vmap
from jax.tree_util import tree_leaves, tree_map, tree_structure

import dynamax.ssm as ssm_module
from dynamax.hidden_markov_model import GaussianHMM, LogisticRegressionHMM, PoissonHMM
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.ssm import SSM
from dynamax.utils.utils import ensure_array_has_batch_dim


def _assert_trees_allclose(actual, expected, **kwargs):
    assert tree_structure(actual) == tree_structure(expected)
    for actual_leaf, expected_leaf in zip(tree_leaves(actual), tree_leaves(expected)):
        assert jnp.allclose(actual_leaf, expected_leaf, **kwargs)


def _make_gaussian_hmm_example():
    model = GaussianHMM(num_states=2, emission_dim=1)
    key_params, key_sample = jr.split(jr.PRNGKey(0))
    params, props = model.initialize(key_params)
    _, emissions = model.sample(params, key_sample, num_timesteps=12)
    return model, params, props, emissions, None


def _make_poisson_hmm_example():
    model = PoissonHMM(num_states=2, emission_dim=1)
    key_params, key_sample = jr.split(jr.PRNGKey(3))
    params, props = model.initialize(key_params)
    _, emissions = model.sample(params, key_sample, num_timesteps=12)
    return model, params, props, emissions, None


def _make_logistic_regression_hmm_example():
    model = LogisticRegressionHMM(
        num_states=2, input_dim=2, m_step_num_iters=2)
    key_params, key_inputs, key_sample = jr.split(jr.PRNGKey(1), 3)
    params, props = model.initialize(key_params)
    inputs = jr.normal(key_inputs, (12, 2))
    _, emissions = model.sample(
        params, key_sample, num_timesteps=inputs.shape[0], inputs=inputs)
    return model, params, props, emissions, inputs


def _make_lgssm_example():
    model = LinearGaussianSSM(state_dim=2, emission_dim=2)
    key_params, key_samples = jr.split(jr.PRNGKey(2))
    params, props = model.initialize(key_params)
    sample_keys = jr.split(key_samples, 3)
    emissions = vmap(
        lambda key: model.sample(params, key, num_timesteps=15)[1]
    )(sample_keys)
    return model, params, props, emissions, None


def _fit_em_python_reference(model, params, props, emissions, inputs, num_iters):
    """The legacy Python loop, independent of either fit_em execution path."""
    batch_emissions = ensure_array_has_batch_dim(emissions, model.emission_shape)
    batch_inputs = ensure_array_has_batch_dim(inputs, model.inputs_shape)

    @jit
    def em_step(current_params, m_step_state):
        batch_stats, lls = vmap(partial(model.e_step, current_params))(
            batch_emissions, batch_inputs)
        lp = model.log_prior(current_params) + lls.sum()
        next_params, next_state = model.m_step(
            current_params, props, batch_stats, m_step_state)
        return next_params, next_state, lp

    m_step_state = model.initialize_m_step_state(params, props)
    log_probs = []
    for _ in range(num_iters):
        params, m_step_state, lp = em_step(params, m_step_state)
        log_probs.append(lp)
    return params, jnp.array(log_probs)


@pytest.fixture(
    scope="module",
    params=[
        _make_gaussian_hmm_example,
        _make_poisson_hmm_example,
        _make_logistic_regression_hmm_example,
        _make_lgssm_example,
    ],
    ids=["gaussian_hmm", "poisson_hmm", "logistic_regression_hmm", "lgssm"],
)
def em_python_reference(request):
    example = request.param()
    reference = _fit_em_python_reference(*example, num_iters=5)
    return example, reference


@pytest.mark.parametrize(
    "verbose, print_every",
    [(False, 2), (True, 1), (True, 2), (True, 8)],
    ids=["silent", "every_iteration", "partial_final_chunk", "single_chunk"],
)
def test_fit_em_scan_matches_python_loop(em_python_reference, verbose, print_every):
    """Full and chunked scans preserve every log probability and optimizer state."""
    example, (loop_params, loop_lps) = em_python_reference
    model, params, props, emissions, inputs = example

    scan_params, scan_lps = model.fit_em(
        params, props, emissions, inputs=inputs, num_iters=5,
        verbose=verbose, print_every=print_every)

    assert scan_lps.shape == loop_lps.shape == (5,)
    assert jnp.allclose(scan_lps, loop_lps, rtol=1e-5, atol=1e-5)
    _assert_trees_allclose(scan_params, loop_params, rtol=1e-5, atol=1e-5)


def test_fit_em_scan_supports_outer_jit_vmap():
    """A vmapped fold fit can compile as one program without unrolling EM."""
    model, params, props, emissions, _ = _make_gaussian_hmm_example()
    batched_emissions = jnp.stack((emissions, emissions))

    def fit_one(sequence):
        return model.fit_em(
            params, props, sequence, num_iters=2, verbose=False)

    batched_params, batched_lps = jit(vmap(fit_one))(batched_emissions)
    single_params, single_lps = fit_one(emissions)

    assert batched_lps.shape == (2, 2)
    assert jnp.allclose(batched_lps[0], single_lps, rtol=1e-5, atol=1e-5)
    _assert_trees_allclose(
        tree_map(lambda x: x[0], batched_params), single_params,
        rtol=1e-5, atol=1e-5)


class _CountingSSM(SSM):
    """A tiny model whose history exposes lost or reordered optimizer state."""

    @property
    def emission_shape(self):
        return (1,)

    def initial_distribution(self, params, inputs=None):
        raise NotImplementedError

    def transition_distribution(self, params, state, inputs=None):
        raise NotImplementedError

    def emission_distribution(self, params, state, inputs=None):
        raise NotImplementedError

    def initialize_m_step_state(self, params, props):
        return jnp.array(0)

    def e_step(self, params, emissions, inputs=None):
        return emissions.sum(), params

    def m_step(self, params, props, batch_stats, m_step_state):
        return params + m_step_state + 1, m_step_state + 1


@pytest.mark.parametrize(
    "print_every, expected_updates, chunk_lengths",
    [(1, [0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1]),
     (2, [0, 2, 4, 5], [2, 2, 1]),
     (8, [0, 5], [5])],
)
def test_fit_em_progress_tracks_dispatched_chunks(
        monkeypatch, print_every, expected_updates, chunk_lengths):
    """Progress follows chunk dispatch without forcing device synchronization."""
    events = []
    bar_totals = []
    dispatched_results = []

    class ProgressSpy:
        def __init__(self, iterable, **kwargs):
            bar_totals.append(len(iterable))

        def update(self, dispatched):
            events.append(("update", dispatched))

        def __iter__(self):
            pytest.fail("EM should update progress explicitly after each chunk")

        def on_interrupt(self):
            events.append(("interrupt",))

    def track_dispatch(fun, **kwargs):
        compiled = jit(fun, **kwargs)

        def run_chunk(carry, length):
            result = compiled(carry, length)
            dispatched_results.append(result)
            events.append(("dispatch", length))
            return result

        return run_chunk

    def unexpected_synchronization(*args, **kwargs):
        pytest.fail("progress reporting should not force device synchronization")

    monkeypatch.setattr(ssm_module, "progress_bar", ProgressSpy)
    monkeypatch.setattr(ssm_module, "jit", track_dispatch)
    monkeypatch.setattr(
        ssm_module, "block_until_ready", unexpected_synchronization, raising=False)
    fitted_params, lps = _CountingSSM().fit_em(
        jnp.array(0.0), None, jnp.ones((2, 1)), num_iters=5,
        verbose=True, print_every=print_every)

    assert bar_totals == [5]
    expected_events = [("update", 0)]
    for length, dispatched in zip(chunk_lengths, expected_updates[1:]):
        expected_events.extend([("dispatch", length), ("update", dispatched)])
    assert events == expected_events
    assert len(dispatched_results) == len(chunk_lengths)
    for result, length, dispatched in zip(
            dispatched_results, chunk_lengths, expected_updates[1:]):
        (chunk_params, chunk_state), chunk_lps = result
        assert chunk_params == dispatched * (dispatched + 1) / 2
        assert chunk_state == dispatched
        assert chunk_lps.shape == (length,)
    assert fitted_params == 15
    assert jnp.array_equal(lps, jnp.array([0.0, 1.0, 3.0, 6.0, 10.0]))


@pytest.mark.parametrize("print_every", [0, -1, 1.5, "2", None, True])
def test_fit_em_silent_ignores_progress_interval(monkeypatch, print_every):
    """Silent fitting neither creates progress nor forces device synchronization."""
    def unexpected_call(*args, **kwargs):
        pytest.fail("silent fitting should not use progress or explicit synchronization")

    monkeypatch.setattr(ssm_module, "progress_bar", unexpected_call)
    monkeypatch.setattr(ssm_module, "block_until_ready", unexpected_call, raising=False)
    fitted_params, lps = _CountingSSM().fit_em(
        jnp.array(0.0), None, jnp.ones((2, 1)), num_iters=5,
        verbose=False, print_every=print_every)

    assert fitted_params == 15
    assert jnp.array_equal(lps, jnp.array([0.0, 1.0, 3.0, 6.0, 10.0]))


@pytest.mark.parametrize("print_every", [0, -1, 1.5, "2", None, True])
def test_fit_em_verbose_rejects_invalid_progress_interval(monkeypatch, print_every):
    def unexpected_progress(*args, **kwargs):
        pytest.fail("invalid intervals should fail before creating a progress bar")

    monkeypatch.setattr(ssm_module, "progress_bar", unexpected_progress)
    with pytest.raises(ValueError, match="print_every"):
        _CountingSSM().fit_em(
            jnp.array(0.0), None, jnp.ones((2, 1)), num_iters=5,
            verbose=True, print_every=print_every)


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("num_iters", [0, -1])
def test_fit_em_zero_iters_returns_inputs_unchanged(monkeypatch, verbose, num_iters):
    """An empty fit returns without tracing an EM step or creating progress."""
    model = _CountingSSM()
    params = jnp.array(0.0)

    def unexpected_call(*args, **kwargs):
        pytest.fail("an empty fit should not run or trace an EM step")

    monkeypatch.setattr(model, "e_step", unexpected_call)
    monkeypatch.setattr(model, "m_step", unexpected_call)
    monkeypatch.setattr(ssm_module, "progress_bar", unexpected_call)
    monkeypatch.setattr(ssm_module, "block_until_ready", unexpected_call, raising=False)

    fitted_params, lps = model.fit_em(
        params, None, jnp.ones((2, 1)), num_iters=num_iters, verbose=verbose)

    assert lps.shape == (0,)
    assert fitted_params is params


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_fit_em_progress_closes_when_chunk_fails(monkeypatch, error_type):
    """A synchronous chunk failure or interruption closes the progress bar."""
    events = []
    model = _CountingSSM()

    class ProgressSpy:
        def __init__(self, iterable, **kwargs):
            pass

        def update(self, dispatched):
            events.append(("update", dispatched))

        def on_interrupt(self):
            events.append(("interrupt",))

    def fail_e_step(*args, **kwargs):
        raise error_type("EM step tracing failed")

    monkeypatch.setattr(ssm_module, "progress_bar", ProgressSpy)
    monkeypatch.setattr(model, "e_step", fail_e_step)
    with pytest.raises(error_type, match="EM step tracing failed"):
        model.fit_em(
            jnp.array(0.0), None, jnp.ones((2, 1)), num_iters=5,
            verbose=True, print_every=2)

    assert events == [("update", 0), ("interrupt",)]
