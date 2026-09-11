"""Fast leave-one-out cross-validation for Gaussian HMM state counts.

The fitting program pads every candidate model to the largest candidate state
count. Inactive states are masked out, which gives every candidate the same
parameter shapes and allows JAX to batch candidates and folds in one compiled
program.

This module currently supports :class:`GaussianHMM` only. Other HMM emission
families can require different parameter padding or masked M-step behavior.
"""

from functools import partial
from typing import Any, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, tree_util, vmap
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.gaussian_hmm import GaussianHMM


# Padded concentrations must be greater than one so the Dirichlet modes used by
# the standard HMM M-step are finite. The corresponding probability mass is
# removed immediately after every M-step.
_PAD_CONCENTRATION = 1.1


class _PaddedSetup(NamedTuple):
    """Arrays shared by the random-start and explicit-start compiled programs."""

    masks: Array
    initial_concentrations: Array
    transition_concentrations: Array
    emission_prior_means: Array
    emission_prior_concentrations: Array
    emission_prior_scales: Array
    emission_prior_extra_dfs: Array
    params: Any
    props: Any


def _project_markov_params(initial_probs: Array, transition_matrix: Array, state_mask: Array) -> Tuple[Array, Array]:
    """Make the inactive states unreachable while keeping valid probabilities."""
    initial_probs = jnp.where(state_mask, initial_probs, 0.0)
    initial_probs = initial_probs / initial_probs.sum(axis=-1, keepdims=True)

    transition_matrix = jnp.where(state_mask[None, :], transition_matrix, 0.0)
    row_sums = transition_matrix.sum(axis=-1, keepdims=True)
    transition_matrix = transition_matrix / jnp.where(row_sums > 0, row_sums, 1.0)
    transition_matrix = jnp.where(
        state_mask[:, None],
        transition_matrix,
        jnp.eye(state_mask.shape[0], dtype=transition_matrix.dtype),
    )
    return initial_probs, transition_matrix


def _project_params(params: Any, state_mask: Array) -> Any:
    """Project a Gaussian HMM parameter set onto its active states."""
    initial_probs, transition_matrix = _project_markov_params(
        params.initial.probs, params.transitions.transition_matrix, state_mask
    )
    return params._replace(
        initial=params.initial._replace(probs=initial_probs),
        transitions=params.transitions._replace(transition_matrix=transition_matrix),
    )


class _PaddedGaussianHMM(GaussianHMM):
    """A Gaussian HMM whose trailing, padded states are inert."""

    def __init__(self, state_mask: Array, emission_dim: int, **kwargs: Any):
        super().__init__(num_states=state_mask.shape[0], emission_dim=emission_dim, **kwargs)
        self.state_mask = state_mask

    def _inference_args(self, params: Any, emissions: Array, inputs: Optional[Array] = None) -> Tuple:
        initial_probs, transition_matrix, log_likelihoods = super()._inference_args(params, emissions, inputs)
        initial_probs, transition_matrix = _project_markov_params(
            initial_probs, transition_matrix, self.state_mask
        )
        log_likelihoods = jnp.where(self.state_mask[None, :], log_likelihoods, -jnp.inf)
        return initial_probs, transition_matrix, log_likelihoods

    def m_step(self, params: Any, props: Any, batch_stats: Any, m_step_state: Any) -> Tuple[Any, Any]:
        params, m_step_state = super().m_step(params, props, batch_stats, m_step_state)
        return _project_params(params, self.state_mask), m_step_state


def _leave_one_out_folds(values: Array) -> Array:
    """Materialize one leave-one-out training set per leading-axis element."""
    num_batches = values.shape[0]
    base_indices = jnp.arange(num_batches - 1)[None, :]
    held_out_indices = jnp.arange(num_batches)[:, None]
    train_indices = base_indices + (base_indices >= held_out_indices)
    return values[train_indices]


def _pad_params(params: Any, max_num_states: int) -> Any:
    """Pad a Gaussian HMM parameter set with valid trailing-state parameters."""
    num_states, emission_dim = params.emissions.means.shape

    initial_probs = np.zeros(max_num_states, dtype=np.asarray(params.initial.probs).dtype)
    initial_probs[:num_states] = np.asarray(params.initial.probs)

    transition_matrix = np.eye(max_num_states, dtype=np.asarray(params.transitions.transition_matrix).dtype)
    transition_matrix[:num_states, :] = 0.0
    transition_matrix[:num_states, :num_states] = np.asarray(params.transitions.transition_matrix)

    unpadded_means = np.asarray(params.emissions.means)
    unpadded_covs = np.asarray(params.emissions.covs)
    means = np.repeat(unpadded_means[:1], max_num_states, axis=0)
    means[:num_states] = unpadded_means
    covs = np.repeat(unpadded_covs[:1], max_num_states, axis=0)
    covs[:num_states] = unpadded_covs

    return params._replace(
        initial=params.initial._replace(probs=jnp.asarray(initial_probs)),
        transitions=params.transitions._replace(transition_matrix=jnp.asarray(transition_matrix)),
        emissions=params.emissions._replace(means=jnp.asarray(means), covs=jnp.asarray(covs)),
    )


def _prepare_models(
    models: Sequence[GaussianHMM],
    emissions: Array,
    key: Optional[Array] = None,
    initialization_method: str = "kmeans",
    initial_params: Optional[Sequence[Tuple[Any, Any]]] = None,
) -> _PaddedSetup:
    """Initialize emissions and assemble equal-shaped parameters and priors."""
    if not models:
        raise ValueError("`models` must contain at least one GaussianHMM.")
    if any(type(model) is not GaussianHMM for model in models):
        raise TypeError("`cross_validate_num_states` currently supports ordinary GaussianHMM models only.")
    if any(model.num_states < 1 for model in models):
        raise ValueError("Every candidate must have at least one state.")

    emissions = jnp.asarray(emissions)
    emission_dim = models[0].emission_dim
    if emissions.ndim != 3 or emissions.shape[0] < 2:
        raise ValueError("`emissions` must contain at least two batched sequences with shape (batch, time, dim).")
    if emissions.shape[-1] != emission_dim or any(model.emission_dim != emission_dim for model in models):
        raise ValueError("All models and `emissions` must have the same emission dimension.")

    num_models = len(models)
    max_num_states = max(model.num_states for model in models)
    if initial_params is not None and len(initial_params) != num_models:
        raise ValueError("`initial_params` must contain one (params, props) pair per model.")
    if initial_params is None and key is None:
        raise ValueError("`key` is required when `initial_params` is not supplied.")

    if initial_params is None:
        keys = jr.split(key, num_models)
        initialization_emissions = np.asarray(emissions) if initialization_method.lower() == "kmeans" else emissions
        initialized = []
        for model, model_key in zip(models, keys):
            num_states = model.num_states
            initialized.append(
                model.initialize(
                    key=model_key,
                    method=initialization_method,
                    initial_probs=jnp.ones(num_states) / num_states,
                    transition_matrix=jnp.ones((num_states, num_states)) / num_states,
                    emissions=initialization_emissions,
                )
            )
    else:
        initialized = list(initial_params)

    masks = []
    initial_concentrations = []
    transition_concentrations = []
    emission_prior_means = []
    emission_prior_concentrations = []
    emission_prior_scales = []
    emission_prior_extra_dfs = []
    padded_params = []

    for model, (params, _) in zip(models, initialized):
        num_states = model.num_states
        if params.initial.probs.shape != (num_states,):
            raise ValueError("Each initial parameter set must match its model's number of states.")

        mask = np.arange(max_num_states) < num_states
        initial_concentration = np.full(
            max_num_states,
            _PAD_CONCENTRATION,
            dtype=np.asarray(model.initial_component.initial_probs_concentration).dtype,
        )
        initial_concentration[:num_states] = np.asarray(model.initial_component.initial_probs_concentration)

        transition_concentration = np.full(
            (max_num_states, max_num_states),
            _PAD_CONCENTRATION,
            dtype=np.asarray(model.transition_component.concentration).dtype,
        )
        transition_concentration[:num_states, :num_states] = np.asarray(model.transition_component.concentration)

        emission_component = model.emission_component
        masks.append(mask)
        initial_concentrations.append(initial_concentration)
        transition_concentrations.append(transition_concentration)
        emission_prior_means.append(np.asarray(emission_component.emission_prior_mean))
        emission_prior_concentrations.append(np.asarray(emission_component.emission_prior_conc))
        emission_prior_scales.append(np.asarray(emission_component.emission_prior_scale))
        emission_prior_extra_dfs.append(np.asarray(emission_component.emission_prior_df - emission_dim))
        padded_params.append(_pad_params(params, max_num_states))

    def stack(values):
        return jnp.asarray(np.stack(values))
    params = tree_util.tree_map(lambda *xs: stack(xs), *padded_params)
    return _PaddedSetup(
        masks=stack(masks),
        initial_concentrations=stack(initial_concentrations),
        transition_concentrations=stack(transition_concentrations),
        emission_prior_means=stack(emission_prior_means),
        emission_prior_concentrations=stack(emission_prior_concentrations),
        emission_prior_scales=stack(emission_prior_scales),
        emission_prior_extra_dfs=stack(emission_prior_extra_dfs),
        params=params,
        props=initialized[0][1],
    )


def _fit_and_score(setup: _PaddedSetup, train_emissions: Array, val_emissions: Array, num_iters: int) -> Array:
    """Batch candidate models and validation folds; called inside one outer JIT."""

    def _fit_one_model(
        state_mask,
        initial_concentration,
        transition_concentration,
        emission_prior_mean,
        emission_prior_concentration,
        emission_prior_scale,
        emission_prior_extra_df,
        params,
    ):
        model = _PaddedGaussianHMM(
            state_mask=state_mask,
            emission_dim=params.emissions.means.shape[-1],
            initial_probs_concentration=initial_concentration,
            transition_matrix_concentration=transition_concentration,
            transition_matrix_stickiness=0.0,
            emission_prior_mean=emission_prior_mean,
            emission_prior_concentration=emission_prior_concentration,
            emission_prior_scale=emission_prior_scale,
            emission_prior_extra_df=emission_prior_extra_df,
        )
        params = _project_params(params, state_mask)

        def _fit_one_fold(fold_emissions, held_out_emissions):
            fit_params, _ = model.fit_em(
                params,
                setup.props,
                fold_emissions,
                num_iters=num_iters,
                verbose=False,
            )
            return model.marginal_log_prob(fit_params, held_out_emissions)

        return vmap(_fit_one_fold)(train_emissions, val_emissions)

    return vmap(_fit_one_model)(
        setup.masks,
        setup.initial_concentrations,
        setup.transition_concentrations,
        setup.emission_prior_means,
        setup.emission_prior_concentrations,
        setup.emission_prior_scales,
        setup.emission_prior_extra_dfs,
        setup.params,
    )


@partial(jit, static_argnames=("num_iters",))
def _cross_validate_random_start(
    key: Array,
    setup: _PaddedSetup,
    emissions: Array,
    num_iters: int,
) -> Array:
    """Sample all Markov parameters, fit, and score in one compiled program."""
    initial_key, transition_key = jr.split(key)
    initial_probs = jr.dirichlet(
        initial_key,
        setup.initial_concentrations,
        dtype=setup.initial_concentrations.dtype,
    )
    transition_matrices = jr.dirichlet(
        transition_key,
        setup.transition_concentrations,
        dtype=setup.transition_concentrations.dtype,
    )
    setup = setup._replace(
        params=setup.params._replace(
            initial=setup.params.initial._replace(probs=initial_probs),
            transitions=setup.params.transitions._replace(transition_matrix=transition_matrices),
        )
    )
    # Renormalizing the active coordinates of a padded Dirichlet draw has the
    # requested lower-dimensional Dirichlet distribution.
    return _fit_and_score(
        setup,
        _leave_one_out_folds(emissions),
        emissions,
        num_iters,
    )


@partial(jit, static_argnames=("num_iters",))
def _cross_validate_from_params(
    setup: _PaddedSetup,
    emissions: Array,
    num_iters: int,
) -> Array:
    """Fit and score explicit starts; used to verify padding equivalence."""
    return _fit_and_score(
        setup,
        _leave_one_out_folds(emissions),
        emissions,
        num_iters,
    )


def cross_validate_num_states(
    models: Sequence[GaussianHMM],
    emissions: Float[Array, "num_batches num_timesteps emission_dim"],
    key: Array,
    num_iters: int = 100,
    initialization_method: str = "kmeans",
) -> Tuple[Float[Array, " num_models"], Float[Array, "num_models num_batches"]]:
    """Run leave-one-out cross-validation over Gaussian HMM state counts.

    The candidates are padded to the largest ``model.num_states`` and fitted
    with a single JIT-compiled program. Nested :func:`jax.vmap` calls batch both
    candidate models and leave-one-out folds. The initial probabilities and
    transition matrices are sampled together inside that program. Emission
    initialization happens once per candidate on the full training set before
    compilation, matching the initialization used by the original notebook.

    Args:
        models: Candidate ``GaussianHMM`` instances. Their order determines the
            order of the returned scores, and their existing prior settings are
            preserved.
        emissions: Batched sequences with shape ``(batch, time, emission_dim)``.
            Each sequence is held out once.
        key: Random key used for emission and Markov-parameter initialization.
        num_iters: Number of EM iterations per fold.
        initialization_method: Gaussian emission initialization method passed to
            :meth:`GaussianHMM.initialize`, typically ``"kmeans"`` or ``"prior"``.

    Returns:
        The average validation log likelihood for each candidate and the full
        ``(num_models, num_batches)`` table of held-out log likelihoods.

    Note:
        This function intentionally supports only the ordinary, full-covariance
        ``GaussianHMM``. In particular, shared-covariance models are not padding
        equivalent because their M-step depends on the total number of states.

        Padding makes every candidate pay the inference cost of the largest
        state count and batching raises peak memory use. Separate sweeps, or
        groups of nearby state counts, can be faster for large datasets or
        widely separated candidates.
    """
    if num_iters < 0:
        raise ValueError("`num_iters` must be nonnegative.")

    models = tuple(models)
    emissions = jnp.asarray(emissions)
    emission_key, markov_key = jr.split(key)
    setup = _prepare_models(
        models,
        emissions,
        key=emission_key,
        initialization_method=initialization_method,
    )
    val_lls = _cross_validate_random_start(
        markov_key,
        setup,
        emissions,
        num_iters=num_iters,
    )
    return val_lls.mean(axis=1), val_lls
