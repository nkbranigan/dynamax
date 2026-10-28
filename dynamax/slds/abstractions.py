"""
Abstract classes of the components of a switching linear dynamical system (SLDS), the tuples of their
sufficient statistics and parameters, and the base class :class:`SLDS` that combines and fits the components.
"""
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from fastprogress.fastprogress import progress_bar
from jax import jit, lax, vmap
from jax.tree_util import tree_map
from jaxtyping import Array, Float, Int

from dynamax.hidden_markov_model.inference import hmm_posterior_mode
from dynamax.linear_gaussian_ssm.block_tridiag import block_tridiag_mvn_sample
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.slds.inference import (
    ContinuousPosteriorSLDS, DiscretePosteriorSLDS, PosteriorSLDS,
    initialize_posterior, slds_elbo, update_continuous_posterior, update_discrete_posterior)
from dynamax.slds.initial import SLDSInitialState
from dynamax.ssm import SSM
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.utils import ensure_array_has_batch_dim


# ---- sufficient statistics ------------------------------------------------------------------
class DynamicsStats(NamedTuple):
    """Expected sufficient statistics of the dynamics, weighted by the posterior probability of each discrete state.

    The dynamics are a regression of each continuous state on the previous continuous state, the input, and a
    constant (the P regressors). Every time step after the first adds to the sums; the first time step gives
    the moments of the initial continuous state.

    :param Ex0: expected initial continuous state, shape (D,)
    :param Ex0x0T: expected outer product of the initial continuous state, shape (D, D)
    :param sum_w: total posterior probability of each discrete state, shape (K,)
    :param sum_phiphiT: weighted sum of the outer products of the regressors, shape (K, P, P)
    :param sum_phixT: weighted sum of the outer products of the regressors and the continuous state, shape (K, P, D)
    :param sum_xxT: weighted sum of the outer products of the continuous state, shape (K, D, D)
    """
    Ex0: Float[Array, " state_dim"]
    Ex0x0T: Float[Array, "state_dim state_dim"]
    sum_w: Float[Array, " num_states"]
    sum_phiphiT: Float[Array, "num_states num_regressors num_regressors"]
    sum_phixT: Float[Array, "num_states num_regressors state_dim"]
    sum_xxT: Float[Array, "num_states state_dim state_dim"]


class EmissionStats(NamedTuple):
    """Expected sufficient statistics of the emissions, weighted by the posterior probability of each discrete state.

    The emissions are a regression on the continuous state, the input, and a constant (the P regressors). With
    a single subspace all states share one emission model, every weight is one, and the leading axis has size 1.

    :param sum_w: total weight of each emission model, shape (K,) or (1,)
    :param sum_psipsiT: weighted sum of the outer products of the regressors, shape (K, P, P)
    :param sum_psiyT: weighted sum of the outer products of the regressors and the emission, shape (K, P, N)
    :param sum_yyT: weighted sum of the outer products of the emission, shape (K, N, N)
    """
    sum_w: Float[Array, " num_subspaces"]
    sum_psipsiT: Float[Array, "num_subspaces num_regressors num_regressors"]
    sum_psiyT: Float[Array, "num_subspaces num_regressors emission_dim"]
    sum_yyT: Float[Array, "num_subspaces emission_dim emission_dim"]


class RecurrentTransStats(NamedTuple):
    """Statistics of one sequence for the M-step of transitions that depend on the continuous state.

    :param trans_probs: posterior probability of each pair of consecutive discrete states, shape (T-1, K, K)
    :param x_samples: samples of the continuous states from the posterior, shape (S, T, D)
    :param inputs: inputs of the sequence, shape (T, M)
    """
    trans_probs: Float[Array, "num_timesteps_minus_1 num_states num_states"]
    x_samples: Float[Array, "num_samples num_timesteps state_dim"]
    inputs: Float[Array, "num_timesteps input_dim"]


class SuffStatsSLDS(NamedTuple):
    """Expected sufficient statistics of one sequence.

    :param initial: posterior probability of each discrete state at the first time step, shape (K,)
    :param transitions: statistics of the transition component
    :param dynamics: sufficient statistics of the dynamics component
    :param emissions: sufficient statistics of the emission component
    """
    initial: Float[Array, " num_states"]
    transitions: Any
    dynamics: DynamicsStats
    emissions: EmissionStats


# ---- abstract classes -----------------------------------------------------------------------
class SLDSTransitions(ABC):
    """Abstract class for SLDS transitions.

    The next discrete state may depend on the current discrete state, the current continuous state, and the
    next input. Subclasses whose transitions depend on the continuous state set the class attribute
    ``depends_on_x`` to True; the model then uses samples of the continuous states to update the discrete
    posterior, and updates the continuous posterior iteratively instead of in one step.
    """
    depends_on_x: bool = False

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the parameters and their corresponding properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the parameters."""

    @abstractmethod
    def transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Compute the transition matrices of the discrete state.

        The shape is (K, K) when the transitions are stationary, (T-1, K, K) otherwise, with one matrix
        per pair of consecutive time steps.
        """

    @abstractmethod
    def log_transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Compute the log transition matrices, with normalized rows, in the same shape as
        :meth:`transition_matrices`."""

    @abstractmethod
    def transition_row(self, params: ParameterSet, state: Array, x: Array, next_input: Array) -> Array:
        """Compute p(z_{t+1} | z_t, x_t), shape (K,), used for sampling."""

    @abstractmethod
    def expected_log_trans_prob(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Scalar:
        """Compute sum_t E_q(z)[log p(z_{t+1} | z_t, x_t)] at the continuous states ``x``."""

    def neg_hessian_x(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Array:
        """Compute the negative Hessian of :meth:`expected_log_trans_prob` with respect to each continuous state.

        The shape is (T-1, D, D). It is added to the first T-1 diagonal blocks of the precision of the
        continuous posterior.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, qz: DiscretePosteriorSLDS, x_samples: Array, inputs: Array) -> Any:
        """Collect the statistics of one sequence that the M-step needs."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """Initialize the state carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: Any, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """Update the parameters from statistics batched over sequences along a leading axis."""


class SLDSDynamics(ABC):
    """Abstract class for SLDS dynamics, including the initial distribution of the continuous state."""

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the parameters and their corresponding properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Compute log p(x_t | x_{t-1}, z_t = k), shape (T, K). Row 0 holds log p(x_1), the same for every k."""

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, qx: ContinuousPosteriorSLDS, inputs: Array) -> Array:
        """Compute E_q(x)[log p(x_t | x_{t-1}, z_t = k)], shape (T, K)."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """Compute E_q(z)[log p(x | z)] at the continuous states ``x``."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, smoothed_probs: Array, inputs: Array) -> Tuple[Array, Array]:
        """Compute the negative Hessian of :meth:`expected_log_joint_x` with respect to the continuous states.

        Returns:
            tuple of the diagonal blocks, shape (T, D, D), and the blocks just below the diagonal, shape (T-1, D, D)
        """

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> DynamicsStats:
        """Compute the expected sufficient statistics of the dynamics under the posterior."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """Initialize the state carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: DynamicsStats, m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParameterSet, Any]:
        """Update the parameters from sufficient statistics batched over sequences along a leading axis."""


class SLDSEmissions(ABC):
    """Abstract class for SLDS emissions.

    With ``single_subspace`` True, all discrete states share one emission model, and the methods return one
    column of per-state values instead of K.
    """
    single_subspace: bool = True

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        """Shape of one emission."""

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the parameters and their corresponding properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, emissions: Array, x: Array, inputs: Array) -> Array:
        """Compute log p(y_t | x_t, z_t = k), shape (T, K), or (T, 1) with a single subspace."""

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, emissions: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> Array:
        """Compute E_q(x)[log p(y_t | x_t, z_t = k)], shape (T, K), or (T, 1) with a single subspace."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """Compute E_q(z)[log p(y | x, z)] at the continuous states ``x``."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """Compute the negative Hessian of :meth:`expected_log_joint_x` with respect to each continuous state,
        shape (T, D, D)."""

    @abstractmethod
    def invert(self, params: ParameterSet, emissions: Array, inputs: Array) -> Array:
        """Estimate the continuous states from the emissions, to center the starting continuous posterior."""

    @abstractmethod
    def state_weights(self, smoothed_probs: Array) -> Array:
        """Compute the weight of each emission model at each time step.

        The weights are ``smoothed_probs``, shape (T, K), or a column of ones, shape (T, 1), with a single subspace.
        """

    @abstractmethod
    def expected_emissions(self, params: ParameterSet, means: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """Compute the expected emissions at the continuous states ``means``, averaged over the emission models
        with :meth:`state_weights`, shape (T, N)."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, emissions: Array, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> EmissionStats:
        """Compute the expected sufficient statistics of the emissions under the posterior."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """Initialize the state carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: EmissionStats, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """Update the parameters from sufficient statistics batched over sequences along a leading axis."""


# ---- parameters ----------------------------------------------------------------------------
class ParamsSLDS(NamedTuple):
    """Parameters of an SLDS.

    :param initial: initial distribution parameters
    :param transitions: transition distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters
    """
    initial: Any
    transitions: Any
    dynamics: Any
    emissions: Any


# ---- model ---------------------------------------------------------------------------------
def _damp_leaf(old: Array, new: Array, prop: ParameterProperties, alpha: float) -> Array:
    """Mix the old and new values of one parameter, with weight ``alpha`` on the old value."""
    if not prop.trainable:
        return new
    if isinstance(prop.constrainer, tfb.SoftmaxCentered):
        # in log space, so that exact zeros stay zero
        return jax.nn.softmax(alpha * jnp.log(old) + (1 - alpha) * jnp.log(new), axis=-1)
    if prop.constrainer is None:
        return alpha * old + (1 - alpha) * new
    mixed = alpha * prop.constrainer.inverse(old) + (1 - alpha) * prop.constrainer.inverse(new)
    return prop.constrainer.forward(mixed)


def _damp(params: ParamsSLDS, new_params: ParamsSLDS, props: ParamsSLDS, alpha: float) -> ParamsSLDS:
    """Mix the old and new parameters of the initial, transition, and emission components, with weight
    ``alpha`` on the old values. The dynamics always take the new values, as in ssm."""
    if alpha == 0.0:
        return new_params

    def _mix(component_old, component_new, component_props):
        is_leaf = lambda node: isinstance(node, ParameterProperties)
        return tree_map(lambda o, n, p: _damp_leaf(o, n, p, alpha), component_old, component_new, component_props, is_leaf=is_leaf)

    return ParamsSLDS(initial=_mix(params.initial, new_params.initial, props.initial),
                      transitions=_mix(params.transitions, new_params.transitions, props.transitions),
                      dynamics=new_params.dynamics,
                      emissions=_mix(params.emissions, new_params.emissions, props.emissions))


class SLDS(SSM):
    """Base class of switching linear dynamical systems (SLDS).

    An SLDS has a continuous state and a discrete state. At each time step the discrete state picks which
    of $K$ linear dynamical systems the continuous state follows, and the emissions depend on both states.
    The posterior over the states cannot be computed exactly, so the model is fitted with Laplace-EM
    (Zoltowski, Pillow, and Linderman, 2020): the posterior is approximated by the product of a discrete
    posterior and a continuous posterior, and the two are updated in turn.

    The model is built from four components, and its parameters are stored in a :class:`ParamsSLDS` object.
    See :class:`GaussianSLDS` for a concrete model.

    :param num_states: number of discrete states
    :param state_dim: dimensionality of the continuous state
    :param initial_component: object encapsulating the initial distribution of the discrete state
    :param transition_component: object encapsulating the transition distribution of the discrete state
    :param dynamics_component: object encapsulating the dynamics of the continuous state
    :param emission_component: object encapsulating the emission distribution
    :param num_samples: number of samples of the continuous states used to update the discrete
        posterior when the transitions depend on the continuous state
    :param newton_tolerance: convergence tolerance of the continuous posterior update
    :param newton_maxiter: maximum number of iterations of the continuous posterior update
    :param damping: weight on the previous initial, transition, and emission parameters in each M-step,
        between 0 (none) and 1; the dynamics always take the new values
    :param perturb_unused_states: whether to reset the dynamics of a discrete state that explains
        almost no data to a noisy copy of a used state's dynamics
    :param initial_variance: variance of the starting continuous posterior, centered on a
        least-squares estimate of the continuous states from the emissions
    """

    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 initial_component: SLDSInitialState,
                 transition_component: SLDSTransitions,
                 dynamics_component: SLDSDynamics,
                 emission_component: SLDSEmissions,
                 *,
                 num_samples: int = 1,
                 newton_tolerance: float = 1e-4,
                 newton_maxiter: int = 100,
                 damping: float = 0.5,
                 perturb_unused_states: bool = True,
                 initial_variance: float = 0.01):
        self.num_states = num_states
        self.state_dim = state_dim
        self.initial_component = initial_component
        self.transition_component = transition_component
        self.dynamics_component = dynamics_component
        self.emission_component = emission_component
        self.num_samples = num_samples
        self.newton_tolerance = newton_tolerance
        self.newton_maxiter = newton_maxiter
        self.damping = damping
        self.perturb_unused_states = perturb_unused_states
        self.initial_variance = initial_variance

    # ---- shapes and shared helpers ----------------------------------------------------------
    @property
    def emission_shape(self) -> Tuple[int]:
        """Shape of one emission."""
        return self.emission_component.emission_shape

    @property
    def inputs_shape(self) -> Tuple[int]:
        """Shape of one input, ``(0,)`` without inputs."""
        return (self.input_dim,)

    @property
    def input_dim(self) -> int:
        """Input dimension (0 without inputs)."""
        return self.dynamics_component.input_dim

    @property
    def _components(self):
        return self.transition_component, self.dynamics_component, self.emission_component

    def _full_inputs(self, inputs: Optional[Array], num_timesteps: int) -> Array:
        """Return the inputs, or zeros of shape (T, input_dim) when they are None."""
        return jnp.zeros((num_timesteps, self.input_dim)) if inputs is None else inputs

    # ---- unsupported SSM methods ------------------------------------------------------------
    def initial_distribution(self, params, inputs=None):
        raise NotImplementedError("SLDS defines sample() directly; the joint (z, x) distributions are not exposed.")

    def transition_distribution(self, params, state, inputs=None):
        raise NotImplementedError("SLDS defines sample() directly; the joint (z, x) distributions are not exposed.")

    def emission_distribution(self, params, state, inputs=None):
        raise NotImplementedError("SLDS defines sample() directly; the joint (z, x) distributions are not exposed.")

    def log_prob(self, params, states, emissions, inputs=None):
        raise NotImplementedError("log_prob is not implemented for the SLDS.")

    def marginal_log_prob(self, params, emissions, inputs=None):
        raise NotImplementedError("The SLDS marginal likelihood is intractable; use approximate_posterior (ELBO).")

    def fit_sgd(self, *args, **kwargs):
        raise NotImplementedError("The SLDS marginal likelihood is intractable; use fit_em.")

    # ---- parameters -------------------------------------------------------------------------
    def initialize(self,
                   key: PRNGKeyT = jr.PRNGKey(0),
                   method: str = "prior",
                   initial_params: Optional[dict] = None,
                   transition_params: Optional[dict] = None,
                   dynamics_params: Optional[dict] = None,
                   emission_params: Optional[dict] = None) -> Tuple[ParamsSLDS, ParamsSLDS]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key: random number generator
            method: initialization method; only ``"prior"`` is supported
            initial_params: dict of arrays overriding the initial distribution parameters
            transition_params: dict of arrays overriding the transition parameters
            dynamics_params: dict of arrays overriding the dynamics parameters, e.g. ``dict(weights=F)``
            emission_params: dict of arrays overriding the emission parameters

        Returns:
            tuple of parameters and their corresponding properties
        """
        k1, k2, k3, k4 = jr.split(key, 4)
        params_initial, props_initial = self.initial_component.initialize(k1, "prior", **(initial_params or {}))
        params_trans, props_trans = self.transition_component.initialize(k2, "prior", **(transition_params or {}))
        params_dyn, props_dyn = self.dynamics_component.initialize(k3, "prior", **(dynamics_params or {}))
        params_emis, props_emis = self.emission_component.initialize(k4, "prior", **(emission_params or {}))
        params = ParamsSLDS(params_initial, params_trans, params_dyn, params_emis)
        props = ParamsSLDS(props_initial, props_trans, props_dyn, props_emis)
        if method != "prior":
            raise ValueError(f"Unknown initialization method {method!r}")
        return params, props

    def log_prior(self, params: ParamsSLDS) -> Scalar:
        """Sum of the components' log priors."""
        return (self.initial_component.log_prior(params.initial)
                + self.transition_component.log_prior(params.transitions)
                + self.dynamics_component.log_prior(params.dynamics)
                + self.emission_component.log_prior(params.emissions))

    # ---- variational E-step -----------------------------------------------------------------
    def _sample_x(self, key: PRNGKeyT, qx: ContinuousPosteriorSLDS) -> Array:
        """Draw ``num_samples`` samples of the continuous states from the continuous posterior, shape (S, T, D).

        Returns zeros instead when the transitions do not depend on the continuous state, since nothing uses
        the samples then.
        """
        if not self.transition_component.depends_on_x:
            return jnp.zeros((self.num_samples,) + qx.smoothed_means.shape)
        return vmap(lambda k: block_tridiag_mvn_sample(k, qx))(jr.split(key, self.num_samples))

    def _update_posterior(self, params: ParamsSLDS, emissions: Array, inputs: Array, posterior: PosteriorSLDS,
                          key: PRNGKeyT) -> Tuple[PosteriorSLDS, Array, Array]:
        """Update the discrete posterior and then the continuous posterior.

        Returns the new posterior and two independent sets of samples from the new continuous posterior, one
        for the sufficient statistics and one for the ELBO.
        """
        k_pre, k_post, k_elbo = jr.split(key, 3)
        x_pre = self._sample_x(k_pre, posterior.continuous)
        qz = update_discrete_posterior(self, params, posterior.continuous, x_pre, emissions, inputs)
        qx = update_continuous_posterior(self, params, qz, posterior.continuous, emissions, inputs)
        new_posterior = PosteriorSLDS(discrete=qz, continuous=qx)
        return new_posterior, self._sample_x(k_post, qx), self._sample_x(k_elbo, qx)

    def _collect_suff_stats(self, params: ParamsSLDS, emissions: Array, inputs: Array, posterior: PosteriorSLDS, x_samples: Array) -> SuffStatsSLDS:
        """Expected sufficient statistics of one sequence."""
        qz, qx = posterior.discrete, posterior.continuous
        return SuffStatsSLDS(
            initial=qz.smoothed_probs[0],
            transitions=self.transition_component.collect_suff_stats(params.transitions, qz, x_samples, inputs),
            dynamics=self.dynamics_component.collect_suff_stats(params.dynamics, qz.smoothed_probs, qx, inputs),
            emissions=self.emission_component.collect_suff_stats(params.emissions, emissions, qz.smoothed_probs, qx, inputs))

    # ---- M-step -----------------------------------------------------------------------------
    def initialize_m_step_state(self, params: ParamsSLDS, props: ParamsSLDS):
        """Initialize the M-step state of each component."""
        return (self.initial_component.initialize_m_step_state(params.initial, props.initial),
                self.transition_component.initialize_m_step_state(params.transitions, props.transitions),
                self.dynamics_component.initialize_m_step_state(params.dynamics, props.dynamics),
                self.emission_component.initialize_m_step_state(params.emissions, props.emissions))

    def m_step(self, params: ParamsSLDS, props: ParamsSLDS, batch_stats: SuffStatsSLDS, m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParamsSLDS, Any]:
        """Update the parameters of every component from statistics batched over sequences.

        Args:
            params: model parameters
            props: properties specifying which parameters should be learned
            batch_stats: sufficient statistics with a leading batch axis
            m_step_state: M-step state of each component
            key: optional random number generator, used to reset unused states (see ``perturb_unused_states``)

        Returns:
            tuple of new parameters and the new M-step state
        """
        s_init, s_trans, s_dyn, s_emis = m_step_state
        p_init, s_init = self.initial_component.m_step(params.initial, props.initial, batch_stats.initial, s_init)
        p_trans, s_trans = self.transition_component.m_step(params.transitions, props.transitions, batch_stats.transitions, s_trans)
        p_dyn, s_dyn = self.dynamics_component.m_step(params.dynamics, props.dynamics, batch_stats.dynamics, s_dyn, key=key)
        p_emis, s_emis = self.emission_component.m_step(params.emissions, props.emissions, batch_stats.emissions, s_emis)
        return ParamsSLDS(p_init, p_trans, p_dyn, p_emis), (s_init, s_trans, s_dyn, s_emis)

    # ---- fitting ----------------------------------------------------------------------------
    def _batch(self, emissions, inputs):
        """Return the emissions and inputs as batches of sequences. Inputs of ``None`` become zeros."""
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        if inputs is None:
            batch_inputs = jnp.zeros(batch_emissions.shape[:2] + (self.input_dim,))
        else:
            batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)
        return batch_emissions, batch_inputs

    def _run(self, params, props, batch_emissions, batch_inputs, num_iters, key, learn, verbose, print_every):
        """Shared loop of fit_em (learn=True) and approximate_posterior (learn=False)."""
        posterior = vmap(partial(initialize_posterior, self, params))(batch_emissions, batch_inputs)
        m_step_state = self.initialize_m_step_state(params, props) if learn else None

        def _step(carry, _):
            params, m_step_state, posterior, key = carry
            key, k_posterior, k_m = jr.split(key, 3)
            keys = jr.split(k_posterior, batch_emissions.shape[0])
            posterior, x_post, x_elbo = vmap(self._update_posterior, (None, 0, 0, 0, 0))(params, batch_emissions, batch_inputs, posterior, keys)
            elbo = vmap(partial(slds_elbo, self, params))(posterior, batch_emissions, batch_inputs, x_elbo).sum()
            if learn:
                objective = self.log_prior(params) + elbo
                stats = vmap(self._collect_suff_stats, (None, 0, 0, 0, 0))(params, batch_emissions, batch_inputs, posterior, x_post)
                new_params, m_step_state = self.m_step(params, props, stats, m_step_state,
                                                      key=k_m if self.perturb_unused_states else None)
                params = _damp(params, new_params, props, self.damping)
            else:
                objective = elbo
            return (params, m_step_state, posterior, key), objective

        @partial(jit, static_argnums=1)
        def _run_steps(carry, num_steps):
            return lax.scan(_step, carry, None, length=num_steps)

        carry = (params, m_step_state, posterior, key)
        if num_iters <= 0:
            return params, jnp.array([]), posterior
        if not verbose:
            (params, _, posterior, _), objectives = _run_steps(carry, num_iters)
            return params, objectives, posterior
        if not isinstance(print_every, Integral) or print_every < 1:
            raise ValueError(f"print_every must be a positive integer, got {print_every!r}")
        objectives = []
        pbar = progress_bar(range(num_iters))
        pbar.update(0)
        for start in range(0, num_iters, print_every):
            num_steps = min(print_every, num_iters - start)
            carry, chunk = _run_steps(carry, num_steps)
            objectives.append(chunk)
            pbar.update(start + num_steps)
        params, _, posterior, _ = carry
        return params, jnp.concatenate(objectives), posterior

    def fit_em(self,
               params: ParamsSLDS,
               props: ParamsSLDS,
               emissions: Array,
               inputs: Optional[Array] = None,
               num_iters: int = 100,
               key: PRNGKeyT = jr.PRNGKey(0),
               return_posterior: bool = False,
               verbose: bool = True,
               print_every: int = 1):
        """Fit the parameters with Laplace-EM.

        Each iteration updates the approximate posterior over the states (the E-step) and then the
        parameters (the M-step). The objective recorded at each iteration is the evidence lower bound
        (ELBO) on the log likelihood, plus the log prior of the parameters.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        Args:
            params: model parameters, as returned by :meth:`initialize`
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            inputs: one or more sequences of corresponding inputs
            num_iters: number of iterations of EM to run
            key: random number generator
            return_posterior: whether to also return the posterior from the last E-step. It was
                computed under the parameters before the final M-step.
            verbose: whether or not to show a progress bar. Use ``False`` when calling
                ``fit_em`` inside ``jit`` or ``vmap``.
            print_every: number of EM iterations between progress bar updates. Ignored when
                ``verbose=False``.

        Returns:
            tuple of new parameters and objective values over the course of EM iterations, followed
            by the posterior when ``return_posterior=True``
        """
        batch_emissions, batch_inputs = self._batch(emissions, inputs)
        params, log_probs, posterior = self._run(params, props, batch_emissions, batch_inputs, num_iters, key,
                                                 True, verbose, print_every)
        return (params, log_probs, posterior) if return_posterior else (params, log_probs)

    def approximate_posterior(self,
                              params: ParamsSLDS,
                              emissions: Array,
                              inputs: Optional[Array] = None,
                              num_iters: int = 50,
                              key: PRNGKeyT = jr.PRNGKey(0),
                              verbose: bool = False,
                              print_every: int = 1) -> Tuple[PosteriorSLDS, Array]:
        """Compute the approximate posterior over the states for fixed parameters.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        Args:
            params: model parameters
            emissions: one or more sequences of emissions
            inputs: one or more sequences of corresponding inputs
            num_iters: number of posterior updates to run
            key: random number generator
            verbose: whether or not to show a progress bar
            print_every: number of iterations between progress bar updates

        Returns:
            tuple of the posterior, with a leading batch axis even for a single sequence, and the
            ELBO after each iteration
        """
        batch_emissions, batch_inputs = self._batch(emissions, inputs)
        _, elbos, posterior = self._run(params, None, batch_emissions, batch_inputs, num_iters, key,
                                        False, verbose, print_every)
        return posterior, elbos

    # ---- summaries of a posterior -----------------------------------------------------------
    def most_likely_states(self, params: ParamsSLDS, emissions: Array, posterior: PosteriorSLDS,
                           inputs: Optional[Array] = None) -> Int[Array, " num_timesteps"]:
        r"""Compute the most likely sequence of discrete states, $\arg\max_z p(z \mid \hat{x}, y, u)$, where
        $\hat{x}$ are the posterior means of the continuous states.

        Args:
            params: model parameters
            emissions: sequence of emissions
            posterior: approximate posterior, as returned by :meth:`approximate_posterior`
            inputs: optional sequence of corresponding inputs

        Returns:
            most likely discrete states
        """
        x = posterior.continuous.smoothed_means
        inputs = self._full_inputs(inputs, emissions.shape[0])
        ll = self.dynamics_component.log_likelihoods(params.dynamics, x, inputs)
        if not self.emission_component.single_subspace:
            ll = ll + self.emission_component.log_likelihoods(params.emissions, emissions, x, inputs)
        return hmm_posterior_mode(params.initial.probs, self.transition_component.transition_matrices(params.transitions, x, inputs), ll)

    def expected_emissions(self, params: ParamsSLDS, posterior: PosteriorSLDS, inputs: Optional[Array] = None) -> Array:
        """Compute the expected emissions under the approximate posterior.

        Args:
            params: model parameters
            posterior: approximate posterior, as returned by :meth:`approximate_posterior`
            inputs: optional sequence of corresponding inputs

        Returns:
            expected emissions at each time step
        """
        means = posterior.continuous.smoothed_means
        inputs = self._full_inputs(inputs, means.shape[0])
        return self.emission_component.expected_emissions(params.emissions, means, posterior.discrete.smoothed_probs, inputs)

    # ---- generative model -------------------------------------------------------------------
    def sample(self, params: ParamsSLDS, key: PRNGKeyT, num_timesteps: int, inputs: Optional[Array] = None
               ) -> Tuple[Int[Array, " num_timesteps"], Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]]:
        """Sample discrete states, continuous states, and emissions given parameters and (optionally) inputs.

        Args:
            params: model parameters
            key: random number generator
            num_timesteps: number of timesteps
            inputs: optional sequence of inputs

        Returns:
            tuple of discrete states, continuous states, and emissions
        """
        inputs = self._full_inputs(inputs, num_timesteps)
        dyn, emis = self.dynamics_component, self.emission_component
        key_z0, key_x0, key_y0, key_rest = jr.split(key, 4)
        z0 = tfd.Categorical(probs=params.initial.probs).sample(seed=key_z0)
        x0 = params.dynamics.initial_mean + jnp.linalg.cholesky(params.dynamics.initial_cov) @ jr.normal(key_x0, (self.state_dim,))
        y0 = emis.sample_emission(key_y0, params.emissions, z0, x0, inputs[0])

        def _step(carry, args):
            z_prev, x_prev = carry
            k, u_next = args
            k_z, k_x, k_y = jr.split(k, 3)
            row = self.transition_component.transition_row(params.transitions, z_prev, x_prev, u_next)
            z = tfd.Categorical(probs=row).sample(seed=k_z)
            x = dyn.sample_transition(k_x, params.dynamics, z, x_prev, u_next)
            y = emis.sample_emission(k_y, params.emissions, z, x, u_next)
            return (z, x), (z, x, y)

        _, (zs, xs, ys) = lax.scan(_step, (z0, x0), (jr.split(key_rest, num_timesteps - 1), inputs[1:]))
        return (jnp.concatenate([z0[None], zs]), jnp.concatenate([x0[None], xs]), jnp.concatenate([y0[None], ys]))
