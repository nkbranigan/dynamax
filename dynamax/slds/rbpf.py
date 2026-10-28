"""
Rao-Blackwellized particle filters for switching linear dynamical systems (SLDS).

Each particle carries a sampled sequence of discrete states and a Gaussian over the continuous state.
As in `LinearGaussianSSM`, the first emission comes from the initial state, with no transition before it,
and the transition into time step t uses the input at time step t.
"""
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import lax, vmap, jit
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import NamedTuple, Optional

from dynamax.utils.utils import psd_solve
from dynamax.types import PRNGKeyT

class DiscreteParamsRBPF(NamedTuple):
    """
    Parameters of the discrete states of an SLDS, used by the particle filters.

    :param initial_distribution: initial distribution of the discrete state, shape (K,)
    :param transition_matrix: transition matrix of the discrete state, shape (K, K)
    :param proposal_transition_matrix: transition matrix that `rbpfilter` samples new discrete states from,
        shape (K, K)
    """
    initial_distribution: Float[Array, " num_states"]
    transition_matrix : Float[Array, "num_states num_states"]
    proposal_transition_matrix : Float[Array, "num_states num_states"]

class LGParamsRBPF(NamedTuple):
    """Linear Gaussian parameters of an SLDS, one set per discrete state, used by the particle filters.

    :param initial_mean: mean of the initial continuous state, shape (K, D)
    :param initial_cov: covariance of the initial continuous state, shape (K, D, D)
    :param dynamics_weights: dynamics weights, shape (K, D, D)
    :param dynamics_cov: dynamics covariances, shape (K, D, D)
    :param dynamics_bias: dynamics biases, shape (K, D)
    :param dynamics_input_weights: dynamics input weights, shape (K, D, M)
    :param emission_weights: emission weights, shape (K, N, D)
    :param emission_cov: emission covariances, shape (K, N, N)
    :param emission_bias: emission biases, shape (K, N)
    :param emission_input_weights: emission input weights, shape (K, N, M)
    """
    initial_mean: Float[Array, "num_states state_dim"]
    initial_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_weights: Float[Array, "num_states state_dim state_dim"]
    dynamics_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_bias: Float[Array, "num_states state_dim"]
    dynamics_input_weights: Float[Array, "num_states state_dim input_dim"]
    emission_weights: Float[Array, "num_states emission_dim state_dim"]
    emission_cov: Float[Array, "num_states emission_dim emission_dim"]
    emission_bias: Float[Array, "num_states emission_dim"]
    emission_input_weights: Float[Array, "num_states emission_dim input_dim"]


class ParamsRBPF(NamedTuple):
    """Parameters of an SLDS for the particle filters.

    :param discrete: parameters of the Markov chain over the discrete states
    :param linear_gaussian: parameters of the linear Gaussian model for the continuous states and emissions
    """
    discrete: DiscreteParamsRBPF
    linear_gaussian: LGParamsRBPF

    def initialize(self, num_states, state_dim, emission_dim, input_dim=0):
        """
        Fill in the parameters so that every field has one entry per discrete state.

        A parameter given once, shared by all discrete states, is repeated for each state. The optional
        biases and input weights are set to zero when they are ``None``.

        Args:
            num_states: number of discrete states
            state_dim: dimension of the continuous state
            emission_dim: dimension of the emissions
            input_dim: dimension of the inputs

        Returns:
            parameters with one entry per discrete state in every field

        Raises:
            ValueError: if a parameter has neither the shared shape nor the per-state shape
        """
        params = self.linear_gaussian
        initial_mean = params.initial_mean if params.initial_mean.shape == (num_states, state_dim) else jnp.array([params.initial_mean]*num_states)
        initial_cov = params.initial_cov if params.initial_cov.shape == (num_states, state_dim, state_dim) else jnp.array([params.initial_cov]*num_states)
        dynamics_weights = params.dynamics_weights if params.dynamics_weights.shape == (num_states, state_dim, state_dim) else jnp.array([params.dynamics_weights]*num_states)
        dynamics_cov = params.dynamics_cov if params.dynamics_cov.shape == (num_states, state_dim, state_dim) else jnp.array([params.dynamics_cov]*num_states)
        dynamics_bias = params.dynamics_bias if params.dynamics_bias is not None else jnp.zeros((num_states, state_dim))
        dynamics_bias = dynamics_bias if dynamics_bias.shape == (num_states, state_dim) else jnp.array([dynamics_bias]*num_states)
        dynamics_input_weights = params.dynamics_input_weights if params.dynamics_input_weights is not None else jnp.zeros((num_states, state_dim, input_dim))
        dynamics_input_weights = dynamics_input_weights if dynamics_input_weights.shape == (num_states, state_dim, input_dim) else jnp.array([dynamics_input_weights]*num_states)
        emission_weights = params.emission_weights if params.emission_weights.shape == (num_states, emission_dim, state_dim) else jnp.array([params.emission_weights]*num_states)
        emission_cov = params.emission_cov if params.emission_cov.shape == (num_states, emission_dim, emission_dim) else jnp.array([params.emission_cov]*num_states)
        emission_bias = params.emission_bias if params.emission_bias is not None else jnp.zeros((num_states, emission_dim))
        emission_bias = emission_bias if emission_bias.shape == (num_states, emission_dim) else jnp.array([emission_bias]*num_states)
        emission_input_weights = params.emission_input_weights if params.emission_input_weights is not None else jnp.zeros((num_states, emission_dim, input_dim))
        emission_input_weights = emission_input_weights if emission_input_weights.shape == (num_states, emission_dim, input_dim) else jnp.array([emission_input_weights]*num_states)
        
        initialized = ParamsRBPF(
            discrete=self.discrete,
            linear_gaussian=LGParamsRBPF(
                initial_mean=initial_mean,
                initial_cov=initial_cov,
                dynamics_weights=dynamics_weights,
                dynamics_cov=dynamics_cov,
                dynamics_bias=dynamics_bias,
                dynamics_input_weights=dynamics_input_weights,
                emission_weights=emission_weights,
                emission_cov=emission_cov,
                emission_bias=emission_bias,
                emission_input_weights=emission_input_weights,
            )
        )
        _check_params(initialized)
        return initialized


def _check_params(params: ParamsRBPF) -> None:
    """Raise a ValueError unless every field of `params` has the per-state shape that `initialize` produces.

    Only shapes are read, so the check runs at trace time under `jax.jit`.
    """
    fields = {**params.discrete._asdict(), **params.linear_gaussian._asdict()}
    if any(value is None for value in fields.values()):
        raise ValueError("ParamsRBPF has fields set to None; call ParamsRBPF.initialize first.")
    K = jnp.shape(fields["initial_distribution"])[-1]
    D = jnp.shape(fields["initial_mean"])[-1]
    N = jnp.shape(fields["emission_cov"])[-1]
    M = jnp.shape(fields["dynamics_input_weights"])[-1]
    expected = dict(
        initial_distribution=(K,), transition_matrix=(K, K), proposal_transition_matrix=(K, K),
        initial_mean=(K, D), initial_cov=(K, D, D),
        dynamics_weights=(K, D, D), dynamics_cov=(K, D, D), dynamics_bias=(K, D), dynamics_input_weights=(K, D, M),
        emission_weights=(K, N, D), emission_cov=(K, N, N), emission_bias=(K, N), emission_input_weights=(K, N, M))
    for name, shape in expected.items():
        if jnp.shape(fields[name]) != shape:
            raise ValueError(f"ParamsRBPF field {name} has shape {jnp.shape(fields[name])}, expected {shape}; "
                             "call ParamsRBPF.initialize first.")


def resampling(weights, states, means, covariances, key):
    """Resample particles using the multinomial resampling algorithm."""
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(key, jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    new_states = jnp.take(states, resampled_idx, axis=0)
    filtered_means = jnp.take(means, resampled_idx, axis=0)
    filtered_covs = jnp.take(covariances, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    return weights, new_states, filtered_means, filtered_covs

@partial(jit, static_argnums=(1,))
def optimal_resampling(weights, N, key):
    """Find the threshold for resampling particles using the optimal resampling algorithm of Fearnhead and Clifford (2003).
    Returns inidices of resampled particles and their weights.
    """
    # sort weights
    M = weights.shape[0]
    sorted_weights = jnp.sort(weights)
    sorted_idx = jnp.argsort(weights)

    # compute threshold p and value L of particles to retain
    lower_diag = jnp.triu(jnp.ones((M, M)), k=0).T
    ps = lax.dynamic_slice(lower_diag, (M-N, 0), (N-1, M)) @ sorted_weights / jnp.arange(1,N)
    ps = jnp.flip(ps)
    bounds = vmap(lambda ind: (sorted_weights[M-ind-1], sorted_weights[M-ind]))(jnp.arange(1, N))
    preds = vmap(lambda y,x,z : jnp.logical_and(y < x, x < z))(bounds[0], ps, bounds[1])
    L = jnp.where(preds, jnp.arange(1, N), 0).sum()
    p = jnp.where(L==0, 1/N, ps[L-1])
    
    # resample
    res_weights = jnp.where(sorted_weights < p, sorted_weights, 0.0)
    res_weights = res_weights / res_weights.sum()
    res_idx = jr.choice(key, M, shape=(M,), replace=True, p=res_weights)
    unsort_res_idx = sorted_idx[res_idx]

    final_idx = jnp.where(sorted_weights < p, unsort_res_idx, sorted_idx)
    final_weights = jnp.where(sorted_weights < p, p, sorted_weights)

    return final_idx[M-N:], final_weights[M-N:] / final_weights[M-N:].sum() 

def _predict(state, mu, Sigma, params, u):
    """Predict the Gaussian over the next continuous state under the dynamics of discrete state `state`.

    `u` is the input at the time step being predicted.
    """
    F = params.dynamics_weights[state]
    b = params.dynamics_bias[state]
    B = params.dynamics_input_weights[state]
    Q = params.dynamics_cov[state]
    mu_pred = F @ mu + B @ u + b
    Sigma_pred = F @ Sigma @ F.T + Q
    return mu_pred, Sigma_pred

def _condition_on(state, mu_pred, Sigma_pred, params, u, y):
    """Condition the predicted Gaussian over the continuous state on the emission `y`, under discrete state `state`.

    Returns the log likelihood of `y` given the earlier emissions and this discrete state, and the conditioned
    mean and covariance.
    """
    H = params.emission_weights[state]
    d = params.emission_bias[state]
    D = params.emission_input_weights[state]
    R = params.emission_cov[state]
    S = R + H @ Sigma_pred @ H.T
    K = psd_solve(S, H @ Sigma_pred).T
    mu_y = H @ mu_pred + D @ u + d
    ll = MVN(loc = mu_y, covariance_matrix = S).log_prob(y)
    mu_cond = mu_pred + K @ (y - mu_y)
    Sigma_cond = Sigma_pred - K @ S @ K.T
    return ll, mu_cond, Sigma_cond

def _stack_outputs(initial, rest):
    """Put the particles of the first time step in front of those of the later steps and name the outputs."""
    weights, states, means, covs = tree_map(lambda x0, x: jnp.concatenate((x0[None], x)), initial, rest)
    return {"weights": weights, "states": states, "means": means, "covariances": covs}

def rbpfilter(
    num_particles: int,
    params: ParamsRBPF,
    emissions:  Float[Array, "ntime emission_dim"],
    key: PRNGKeyT = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ess_threshold: float = 0.5
    ):
    """
    Run a Rao-Blackwellized particle filter to approximate the filtering distribution of an SLDS.

    Each particle samples its discrete states, from the initial distribution at the first step and from
    ``proposal_transition_matrix`` after that, and runs a Kalman filter step for the continuous state. The
    weights are corrected for sampling from the proposal instead of the transition matrix. The particles are
    resampled whenever the effective sample size falls below ``ess_threshold * num_particles``.

    The function can be jitted with ``num_particles`` marked static.

    Args:
        num_particles: number of particles
        params: model parameters, as returned by :meth:`ParamsRBPF.initialize`
        emissions: sequence of emissions
        key: random number generator
        inputs: optional sequence of corresponding inputs
        ess_threshold: fraction of ``num_particles`` below which the effective sample size triggers resampling

    Returns:
        dict with the particles after each time step (after resampling, if it ran): "weights" (T, num_particles),
        "states" (T, num_particles), "means" (T, num_particles, state_dim) and
        "covariances" (T, num_particles, state_dim, state_dim).
    """
    _check_params(params)
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, params.linear_gaussian.dynamics_input_weights.shape[-1])) if inputs is None else inputs
    lg_params = params.linear_gaussian
    log_transition = jnp.log(params.discrete.transition_matrix)
    log_proposal = jnp.log(params.discrete.proposal_transition_matrix)

    def _update(weights, log_ratios, states, pred_means, pred_covs, u, y, key):
        """Condition each particle on the emission, reweight it, and resample if the effective sample size is low."""
        lls, filtered_means, filtered_covs = vmap(_condition_on, in_axes=(0, 0, 0, None, None, None))(states, pred_means, pred_covs, lg_params, u, y)
        log_weights = jnp.log(weights) + log_ratios + lls
        weights = jnp.exp(log_weights - logsumexp(log_weights))
        resample_cond = 1.0 / jnp.sum(jnp.square(weights)) < ess_threshold * num_particles
        return lax.cond(resample_cond, partial(resampling, key=key), lambda *args: args,
                        weights, states, filtered_means, filtered_covs)

    def _step(carry, args):
        """Advance the particles by one time step.

        The carry holds the weights (num_particles,), previous discrete states (num_particles,), filtered
        means (num_particles, state_dim), filtered covariances (num_particles, state_dim, state_dim) and the key.
        """
        weights, prev_states, filtered_means, filtered_covs, key = carry
        u, y = args
        key, sample_key, resample_key = jr.split(key, 3)

        # Sample z_t from the proposal; the importance ratio corrects for it not being the transition matrix
        new_states = jr.categorical(sample_key, log_proposal[prev_states])
        log_ratios = log_transition[prev_states, new_states] - log_proposal[prev_states, new_states]

        # Predict x_t given z_t, then condition on y_t, reweight and resample
        pred_means, pred_covs = vmap(_predict, in_axes=(0, 0, 0, None, None))(new_states, filtered_means, filtered_covs, lg_params, u)
        particles = _update(weights, log_ratios, new_states, pred_means, pred_covs, u, y, resample_key)
        return (*particles, key), particles

    # t = 1: sample z_1 from the initial distribution and condition x_1 ~ N(mu_{z_1}, Sigma_{z_1}) on y_1
    key, sample_key, resample_key = jr.split(key, 3)
    initial_states = jr.choice(sample_key, params.discrete.initial_distribution.shape[0], shape=(num_particles,), p=params.discrete.initial_distribution)
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    initial_particles = _update(initial_weights, jnp.zeros(num_particles), initial_states,
                                lg_params.initial_mean[initial_states], lg_params.initial_cov[initial_states],
                                inputs[0], emissions[0], resample_key)

    _, particles = lax.scan(_step, (*initial_particles, key), (inputs[1:], emissions[1:]))
    return _stack_outputs(initial_particles, particles)

def rbpfilter_optimal(
    num_particles: int,
    params: ParamsRBPF,
    emissions:  Float[Array, "ntime emission_dim"],
    key: PRNGKeyT = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ):
    """
    Run a Rao-Blackwellized particle filter with optimal resampling.

    It approximates the filtering distribution of an SLDS. At each time step every particle is extended to every
    possible discrete state, each extension is weighted by how likely the move and the emission are, and
    ``num_particles`` of the extensions are kept using the optimal resampling of Fearnhead and Clifford (2003).

    The function can be jitted with ``num_particles`` marked static.

    Args:
        num_particles: number of particles
        params: model parameters, as returned by :meth:`ParamsRBPF.initialize`
        emissions: sequence of emissions
        key: random number generator
        inputs: optional sequence of corresponding inputs

    Returns:
        dict with the particles after each time step, with the same keys and shapes as :func:`rbpfilter`.
    """
    _check_params(params)
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, params.linear_gaussian.dynamics_input_weights.shape[-1])) if inputs is None else inputs
    lg_params = params.linear_gaussian
    num_states, state_dim = lg_params.initial_mean.shape
    log_transition = jnp.log(params.discrete.transition_matrix)

    def _update(weights, log_trans_probs, pred_means, pred_covs, u, y, key):
        """Condition every extension of every particle on the emission, weight it, and keep num_particles of them.

        log_trans_probs, pred_means and pred_covs have leading axes (num_particles, num_states).
        """
        _condition_all = vmap(vmap(_condition_on, in_axes=(0, 0, 0, None, None, None)), in_axes=(None, 0, 0, None, None, None))
        lls, filtered_means, filtered_covs = _condition_all(jnp.arange(num_states), pred_means, pred_covs, lg_params, u, y)
        log_weights = jnp.log(weights)[:, None] + log_trans_probs + lls
        weights = jnp.exp(log_weights - logsumexp(log_weights))

        # Reshape
        weights = weights.reshape((num_particles * num_states,))
        states = jnp.tile(jnp.arange(num_states), num_particles)
        filtered_means = filtered_means.reshape((num_particles * num_states, state_dim))
        filtered_covs = filtered_covs.reshape((num_particles * num_states, state_dim, state_dim))

        # Optimal resampling
        res_idx, res_weights = optimal_resampling(weights, num_particles, key)
        new_states = jnp.take(states, res_idx, axis=0)
        filtered_means = jnp.take(filtered_means, res_idx, axis=0)
        filtered_covs = jnp.take(filtered_covs, res_idx, axis=0)
        return res_weights, new_states, filtered_means, filtered_covs

    def _step(carry, args):
        """Advance the particles by one time step.

        The carry holds the weights (num_particles,), previous discrete states (num_particles,), filtered
        means (num_particles, state_dim), filtered covariances (num_particles, state_dim, state_dim) and the key.
        """
        weights, prev_states, filtered_means, filtered_covs, key = carry
        u, y = args
        key, resample_key = jr.split(key, 2)

        # Predict x_t for every particle and every value of z_t
        _predict_all = vmap(vmap(_predict, in_axes=(0, None, None, None, None)), in_axes=(None, 0, 0, None, None))
        pred_means, pred_covs = _predict_all(jnp.arange(num_states), filtered_means, filtered_covs, lg_params, u)
        particles = _update(weights, log_transition[prev_states], pred_means, pred_covs, u, y, resample_key)
        return (*particles, key), particles

    # t = 1: extend each particle to every z_1, with x_1 ~ N(mu_{z_1}, Sigma_{z_1}) and no transition
    key, resample_key = jr.split(key, 2)
    initial_weights = jnp.ones(shape=(num_particles,)) / num_particles
    expand = lambda x: jnp.broadcast_to(x, (num_particles,) + x.shape)
    initial_particles = _update(initial_weights, expand(jnp.log(params.discrete.initial_distribution)),
                                expand(lg_params.initial_mean), expand(lg_params.initial_cov),
                                inputs[0], emissions[0], resample_key)

    _, particles = lax.scan(_step, (*initial_particles, key), (inputs[1:], emissions[1:]))
    return _stack_outputs(initial_particles, particles)
