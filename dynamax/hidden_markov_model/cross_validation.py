"""Cross-validation over the number of HMM states with a single compiled program.

Selecting the number of discrete states typically means fitting one model per
candidate ``num_states``. Because XLA specializes every compiled program on the
array shapes it sees, that costs one trace-and-compile per candidate, and the
fits run one after another.

:func:`cross_validate_num_states` avoids both costs by *padding* every candidate
model to the largest number of states, ``K_max``. The extra states are made
inert -- zero initial probability, zero incoming transition probability, and
``-inf`` emission log-likelihood -- so they never receive posterior mass and the
fit of the ``k`` live states is the same as it would be for an unpadded
``k``-state model. With every model sharing one shape, all candidates and all
cross-validation folds are fit by a single ``jit(vmap(...))`` program: one
compile, one dispatch, and fully batched execution.

The same trick applies to the random initialization: a Dirichlet draw is a
normalized vector of independent Gammas, so one padded Gamma sample provides
the initial distribution and transition matrix of every candidate model.
"""
from functools import lru_cache, partial
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Type

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, tree_util, vmap
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.abstractions import HMM

# Padded prior concentrations must exceed 1 so that Dirichlet.mode() in the
# M-step stays finite; 1e-6 survives float32 rounding. Any mass this leaks into
# padded states is removed again by re-masking after every M-step.
_PAD_CONCENTRATION = 1.0 + 1e-6


def _mask_params(params, state_mask: Array):
    """Zero the probability of padded states in the initial and transition parameters."""
    probs = params.initial.probs * state_mask
    probs = probs / probs.sum()
    A = params.transitions.transition_matrix * state_mask[None, :]
    # Padded rows are never reached; make them valid self-loops.
    A = jnp.where(state_mask[:, None], A, jnp.eye(state_mask.shape[0], dtype=A.dtype))
    A = A / A.sum(axis=1, keepdims=True)
    return params._replace(initial=params.initial._replace(probs=probs),
                           transitions=params.transitions._replace(transition_matrix=A))


@lru_cache(maxsize=None)
def _padded_class(model_cls: Type[HMM]) -> Type[HMM]:
    """A subclass of ``model_cls`` in which states outside ``state_mask`` are inert."""

    class PaddedHMM(model_cls):
        def __init__(self, state_mask, **kwargs):
            super().__init__(**kwargs)
            self.state_mask = state_mask

        def _inference_args(self, params, emissions, inputs=None):
            initial_probs, transition_matrix, log_likelihoods = \
                super()._inference_args(params, emissions, inputs)
            log_likelihoods = jnp.where(self.state_mask, log_likelihoods, -jnp.inf)
            return initial_probs, transition_matrix, log_likelihoods

        def m_step(self, params, props, batch_stats, m_step_state):
            params, m_step_state = super().m_step(params, props, batch_stats, m_step_state)
            return _mask_params(params, self.state_mask), m_step_state

    PaddedHMM.__name__ = PaddedHMM.__qualname__ = f"Padded{model_cls.__name__}"
    return PaddedHMM


def _pad_emission_params(emission_params, num_states: int, max_num_states: int):
    """Pad every per-state emission parameter (state axis first) up to ``max_num_states``.

    Padded states copy the parameters of state 0, which are valid for any family.
    """
    def _pad(leaf):
        if leaf.ndim >= 1 and leaf.shape[0] == num_states:
            filler = jnp.repeat(leaf[:1], max_num_states - num_states, axis=0)
            return jnp.concatenate([leaf, filler], axis=0)
        return leaf
    return tree_util.tree_map(_pad, emission_params)


def _pad_initial_and_transitions(params, num_states: int, max_num_states: int):
    """Embed a k-state initial distribution and transition matrix in K_max states; padded states are inert."""
    probs = jnp.zeros(max_num_states, dtype=params.initial.probs.dtype).at[:num_states].set(params.initial.probs)
    A = jnp.eye(max_num_states, dtype=params.transitions.transition_matrix.dtype)
    A = A.at[:num_states, :].set(0.0).at[:num_states, :num_states].set(params.transitions.transition_matrix)
    return params._replace(initial=params.initial._replace(probs=probs),
                           transitions=params.transitions._replace(transition_matrix=A))


class _PaddedSetup(NamedTuple):
    models: list            # unpadded models, one per candidate
    masks: Array            # (n, K_max) bool: live states of each model
    init_conc: Array        # (n, K_max) padded initial-distribution prior concentrations
    trans_conc: Array       # (n, K_max, K_max) padded transition prior concentrations
    draw_conc: Array        # (n, K_max + 1, K_max) concentrations for the padded Gamma draw
    params0: Any            # padded initial params stacked over models (initial/transitions are placeholders)
    props: Any              # parameter properties (shared by all candidates)
    unpadded_emissions: list  # unpadded initial emission params, one per candidate


def _prepare(model_cls, all_num_states, emissions, key, init_method, model_kwargs, init_kwargs,
             initial_params=None):
    """Build the padded priors, masks, and initial parameters for every candidate model."""
    all_num_states = [int(k) for k in all_num_states]
    n, K_max = len(all_num_states), max(all_num_states)
    models = [model_cls(num_states=k, **model_kwargs) for k in all_num_states]
    if emissions.ndim != len(models[0].emission_shape) + 2 or emissions.shape[0] < 2:
        raise ValueError("`emissions` must be batched with at least two sequences (one fold per sequence).")

    masks = np.zeros((n, K_max), dtype=bool)
    init_conc = np.full((n, K_max), _PAD_CONCENTRATION)
    trans_conc = np.full((n, K_max, K_max), _PAD_CONCENTRATION)
    draw_conc = np.ones((n, K_max + 1, K_max))
    for i, (k, model) in enumerate(zip(all_num_states, models)):
        a0 = np.asarray(model.initial_component.initial_probs_concentration)
        A0 = np.asarray(model.transition_component.concentration)
        masks[i, :k] = True
        init_conc[i, :k] = a0
        trans_conc[i, :k, :k] = A0
        trans_conc[i, range(k, K_max), range(k, K_max)] = 2.0   # padded rows: mode is a self-loop
        draw_conc[i, 0, :k] = a0
        draw_conc[i, 1:k + 1, :k] = A0

    # Per-model emission initialization (host side, e.g. k-means). The initial
    # distribution and transition matrix are placeholders here; they are drawn
    # inside the compiled program from one padded Gamma sample.
    if initial_params is not None and len(initial_params) != n:
        raise ValueError("`initial_params` must hold one (params, props) pair per candidate in `all_num_states`.")
    padded, unpadded = [], []
    props = None
    for i, (k, model, subkey) in enumerate(zip(all_num_states, models, jr.split(key, n))):
        if initial_params is not None:
            params, props = initial_params[i]
            if params.initial.probs.shape[0] != k:
                raise ValueError(f"`initial_params[{i}]` has {params.initial.probs.shape[0]} states, expected {k}.")
        else:
            kwargs = dict(init_kwargs)
            if init_method == "kmeans":
                kwargs["emissions"] = emissions
            params, props = model.initialize(key=subkey, method=init_method,
                                             initial_probs=jnp.ones(k) / k,
                                             transition_matrix=jnp.ones((k, k)) / k,
                                             **kwargs)
        unpadded.append(params.emissions)
        params = params._replace(emissions=_pad_emission_params(params.emissions, k, K_max))
        padded.append(_pad_initial_and_transitions(params, k, K_max))
    params0 = tree_util.tree_map(lambda *xs: jnp.stack(xs), *padded)

    return _PaddedSetup(models, jnp.asarray(masks), jnp.asarray(init_conc), jnp.asarray(trans_conc),
                        jnp.asarray(draw_conc), params0, props, unpadded)


def _sample_padded_inits(key, draw_conc, masks):
    """Initial distributions and transition matrices for all models from one Gamma draw."""
    K_max = masks.shape[1]
    g = jr.gamma(key, draw_conc)                                     # (n, K_max + 1, K_max)
    init_probs = g[:, 0] * masks
    init_probs = init_probs / init_probs.sum(axis=1, keepdims=True)
    A = g[:, 1:] * masks[:, None, :]
    A = jnp.where(masks[:, :, None], A, jnp.eye(K_max)[None])
    A = A / A.sum(axis=2, keepdims=True)
    return init_probs, A


def _leave_one_out_folds(batched):
    """Stack the leave-one-out training sets of a batched array; ``None`` passes through."""
    if batched is None:
        return None
    n = batched.shape[0]
    return jnp.stack([jnp.concatenate([batched[:i], batched[i + 1:]]) for i in range(n)])


@partial(jit, static_argnames=("padded_cls", "model_kwargs_items", "num_iters", "sample_inits"))
def _cross_validate_jit(padded_cls, model_kwargs_items, num_iters, sample_inits, key, masks, init_conc,
                        trans_conc, draw_conc, params0, props, train_y, val_y, train_u, val_u):
    """One program for every candidate model and every fold."""
    K_max = masks.shape[1]
    if sample_inits:
        init_probs, trans_mats = _sample_padded_inits(key, draw_conc, masks)
    else:
        init_probs, trans_mats = params0.initial.probs, params0.transitions.transition_matrix

    def fit_one_model(mask, ic, tc, pi0, A0, params_i):
        kwargs = dict(model_kwargs_items)
        kwargs.update(num_states=K_max, initial_probs_concentration=ic,
                      transition_matrix_concentration=tc, transition_matrix_stickiness=0.0)
        model = padded_cls(state_mask=mask, **kwargs)
        params = params_i._replace(initial=params_i.initial._replace(probs=pi0),
                                   transitions=params_i.transitions._replace(transition_matrix=A0))

        def _fit_fold(y_train, y_val, u_train, u_val):
            fit_params, _ = model.fit_em(params, props, y_train, inputs=u_train,
                                         num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val, inputs=u_val)

        in_axes = (0, 0, None if train_u is None else 0, None if val_u is None else 0)
        return vmap(_fit_fold, in_axes=in_axes)(train_y, val_y, train_u, val_u)

    return vmap(fit_one_model)(masks, init_conc, trans_conc, init_probs, trans_mats, params0)


def cross_validate_num_states(
        model_cls: Type[HMM],
        all_num_states: Sequence[int],
        emissions: Float[Array, "num_batches num_timesteps emission_dim"],
        key: Array,
        inputs: Optional[Float[Array, "num_batches num_timesteps input_dim"]] = None,
        num_iters: int = 100,
        init_method: str = "prior",
        model_kwargs: Optional[Dict[str, Any]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        initial_params: Optional[Sequence[Tuple[Any, Any]]] = None,
) -> Tuple[Float[Array, " num_models"], Float[Array, "num_models num_batches"]]:
    r"""Leave-one-out cross-validation over the number of states, compiled once.

    For each ``k`` in ``all_num_states``, a ``model_cls(num_states=k, **model_kwargs)``
    is initialized (emission parameters via ``initialize(method=init_method)``,
    initial distribution and transition matrix drawn from the model's priors),
    fit with EM to every leave-one-out fold of ``emissions``, and scored by the
    marginal log likelihood of the held-out sequence.

    Every candidate is padded to ``max(all_num_states)`` states and the padded
    states are masked out of inference, so the whole sweep -- all models and all
    folds -- is one ``jit(vmap(...))`` program. Compared with fitting one model
    per ``k`` this removes ``len(all_num_states) - 1`` compilations and lets the
    fits run batched. Results match the per-model fits to floating-point precision.

    Args:
        model_cls: an HMM class using the standard initial-state and transition
            components (all built-in HMMs qualify).
        all_num_states: candidate numbers of states.
        emissions: batched emissions; one fold is held out per batch.
        key: PRNG key for the random initialization.
        inputs: optional batched inputs, aligned with ``emissions``.
        num_iters: EM iterations per fit.
        init_method: passed to ``initialize`` for the emission parameters
            (``"kmeans"`` for the Gaussian HMMs, ``"prior"`` otherwise).
        model_kwargs: keyword arguments for ``model_cls`` other than ``num_states``
            (e.g. ``emission_dim``, prior hyperparameters). Values must be hashable.
        init_kwargs: extra keyword arguments for ``initialize``.
        initial_params: optional ``(params, props)`` pairs, one per candidate, as returned
            by ``model.initialize``. When given, the random initialization is skipped and
            these parameters are used (padded), e.g. to run several restarts or to compare
            against fitting the candidates one at a time from the same starting point.

    Returns:
        ``(avg_val_lls, val_lls)``: the mean held-out log likelihood per candidate
        and the full ``(num_models, num_batches)`` table.

    Note:
        Models whose M-step depends explicitly on ``num_states`` (currently
        :class:`SharedCovarianceGaussianHMM`) are not exactly equivalent under
        padding. Emission parameters are assumed to carry the state axis first.
    """
    model_kwargs = dict(model_kwargs or {})
    init_kwargs = dict(init_kwargs or {})
    try:
        model_kwargs_items = tuple(sorted(model_kwargs.items()))
        hash(model_kwargs_items)
    except TypeError as e:
        raise TypeError("`model_kwargs` values must be hashable (they are static under jit).") from e

    key_init, key_draw = jr.split(key)
    setup = _prepare(model_cls, all_num_states, emissions, key_init, init_method, model_kwargs, init_kwargs,
                     initial_params)
    val_lls = _cross_validate_jit(
        _padded_class(model_cls), model_kwargs_items, int(num_iters), initial_params is None, key_draw,
        setup.masks, setup.init_conc, setup.trans_conc, setup.draw_conc, setup.params0, setup.props,
        _leave_one_out_folds(emissions), emissions, _leave_one_out_folds(inputs), inputs)
    return val_lls.mean(axis=1), val_lls
