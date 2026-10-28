"""
Variational inference for switching linear dynamical systems.
"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.special import logsumexp, xlogy
from jax.tree_util import tree_map
from jaxtyping import Array, Float

from dynamax.linear_gaussian_ssm.block_tridiag import (
    BlockTridiagPosterior, BlockTridiagPotentials, block_tridiag_mvn_entropy, block_tridiag_mvn_matvec,
    block_tridiag_mvn_smoother, block_tridiag_mvn_solve)
from dynamax.types import Scalar


class DiscretePosteriorSLDS(NamedTuple):
    """Approximate posterior over the discrete states of an SLDS (a Markov chain).

    :param initial_probs: initial distribution of the chain
    :param log_transition_matrices: log transition matrices of the chain, shape (K, K) when the
        transitions are stationary, (T-1, K, K) otherwise
    :param log_likelihoods: log likelihood of each time step under each discrete state, shape (T, K)
    :param log_normalizer: log normalizer of the chain
    :param smoothed_probs: $q(z_t = k)$, shape (T, K)
    :param trans_probs: $q(z_t = i, z_{t+1} = j)$, summed over time to shape (K, K) when the transitions are
        stationary, (T-1, K, K) otherwise
    """
    initial_probs: Float[Array, " num_states"]
    log_transition_matrices: Array
    log_likelihoods: Float[Array, "num_timesteps num_states"]
    log_normalizer: Scalar
    smoothed_probs: Float[Array, "num_timesteps num_states"]
    trans_probs: Array


ContinuousPosteriorSLDS = BlockTridiagPosterior
"""Approximate posterior over the continuous states of an SLDS (a Gaussian in information form)."""


class PosteriorSLDS(NamedTuple):
    """Approximate posterior over the states of an SLDS, $q(z, x) = q(z) q(x)$.

    :param discrete: posterior over the discrete states
    :param continuous: posterior over the continuous states
    """
    discrete: DiscretePosteriorSLDS
    continuous: ContinuousPosteriorSLDS


def _log_space_forward(initial_probs: Array, log_transition_matrices: Array, log_likelihoods: Array
                       ) -> Tuple[Scalar, Array, Array]:
    """Run the forward pass of an HMM in log space.

    The messages are normalized at every step and their log normalizers summed separately, which keeps
    the precision of the messages when the log likelihoods are very large.

    Returns:
        the log normalizer, the normalized log forward messages, shape (T, K), and the log transition
        matrices broadcast to one per step, shape (T-1, K, K).
    """
    num_timesteps, num_states = log_likelihoods.shape
    log_P_t = (log_transition_matrices if log_transition_matrices.ndim == 3
               else jnp.broadcast_to(log_transition_matrices, (num_timesteps - 1, num_states, num_states)))

    def _forward(carry, args):
        log_alpha, log_Z = carry
        log_P_prev, ll = args
        log_alpha = logsumexp(log_alpha[:, None] + log_P_prev, axis=0) + ll
        c = logsumexp(log_alpha)
        return (log_alpha - c, log_Z + c), log_alpha - c

    log_alpha0 = jnp.log(initial_probs) + log_likelihoods[0]
    c0 = logsumexp(log_alpha0)
    (_, log_normalizer), log_alphas = lax.scan(_forward, (log_alpha0 - c0, c0), (log_P_t, log_likelihoods[1:]))
    return log_normalizer, jnp.concatenate([(log_alpha0 - c0)[None], log_alphas]), log_P_t


def log_space_hmm_smoother(initial_probs: Array, log_transition_matrices: Array, log_likelihoods: Array
                           ) -> Tuple[Scalar, Array, Array]:
    """Run the forward-backward algorithm of an HMM in log space.

    Unlike ``hmm_smoother``, it works with log probabilities throughout. With ordinary probabilities, float32
    rounds very small numbers to exactly zero. The pass then fails at any step where the data strongly favor
    only one state, and that state's probability of being entered has been rounded to zero: then
    sum_k p(y_t, z_t = k | y_{1:t-1}) = 0 (division by zero).

    Args:
        initial_probs: initial distribution, shape (K,)
        log_transition_matrices: log transition matrices, shape (K, K) when the transitions are stationary,
            (T-1, K, K) otherwise
        log_likelihoods: log likelihood of each time step under each state, shape (T, K)

    Returns:
        the log normalizer, log p(y_{1:T}); the posterior probability of each state, p(z_t = k | y_{1:T}),
        shape (T, K); and the posterior probability of each pair of consecutive states,
        p(z_t = i, z_{t+1} = j | y_{1:T}), summed over time to shape (K, K) when the transitions are
        stationary, (T-1, K, K) otherwise.
    """
    num_states = log_likelihoods.shape[1]
    time_varying = log_transition_matrices.ndim == 3
    log_normalizer, log_alphas, log_P_t = _log_space_forward(initial_probs, log_transition_matrices, log_likelihoods)

    def _backward(log_beta, args):
        log_P_next, ll_next = args
        log_beta = logsumexp(log_P_next + (ll_next + log_beta)[None, :], axis=1)
        return log_beta - logsumexp(log_beta), log_beta - logsumexp(log_beta)

    _, log_betas = lax.scan(_backward, jnp.zeros(num_states), (log_P_t, log_likelihoods[1:]), reverse=True)
    log_betas = jnp.concatenate([log_betas, jnp.zeros((1, num_states))])
    smoothed_probs = jax.nn.softmax(log_alphas + log_betas, axis=1)
    log_pairs = log_alphas[:-1, :, None] + log_P_t + (log_likelihoods[1:] + log_betas[1:])[:, None, :]
    trans_probs = jnp.exp(log_pairs - logsumexp(log_pairs, axis=(1, 2), keepdims=True))
    if not time_varying:
        trans_probs = trans_probs.sum(0)
    return log_normalizer, smoothed_probs, trans_probs


def _discrete_posterior(initial_probs: Array, log_transition_matrices: Array, log_likelihoods: Array) -> DiscretePosteriorSLDS:
    """Run the log-space smoother and package the result as a discrete posterior."""
    log_normalizer, smoothed_probs, trans_probs = log_space_hmm_smoother(initial_probs, log_transition_matrices, log_likelihoods)
    return DiscretePosteriorSLDS(initial_probs=initial_probs, log_transition_matrices=log_transition_matrices,
                                 log_likelihoods=log_likelihoods, log_normalizer=log_normalizer,
                                 smoothed_probs=smoothed_probs, trans_probs=trans_probs)


def initialize_posterior(model, params, emissions: Array, inputs: Array) -> PosteriorSLDS:
    """Initialize the approximate posterior for one sequence.

    The discrete posterior is uniform. The continuous posterior is a Gaussian centered at the states
    recovered by inverting the emissions, with variance ``model.initial_variance`` in every direction.

    Args:
        model: the SLDS
        params: model parameters
        emissions: sequence of emissions
        inputs: sequence of corresponding inputs

    Returns:
        PosteriorSLDS: the initial posterior.
    """
    num_timesteps = emissions.shape[0]
    K, D = model.num_states, model.state_dim
    time_varying = model.transition_component.depends_on_x
    shape = (num_timesteps - 1, K, K) if time_varying else (K, K)
    qz = _discrete_posterior(jnp.full(K, 1.0 / K), jnp.full(shape, -jnp.log(K)), jnp.zeros((num_timesteps, K)))
    x_hat = model.emission_component.invert(params.emissions, emissions, inputs)
    potentials = BlockTridiagPotentials(J_diag=jnp.tile(jnp.eye(D) / model.initial_variance, (num_timesteps, 1, 1)),
                                        J_lower=jnp.zeros((num_timesteps - 1, D, D)),
                                        h=x_hat / model.initial_variance)
    return PosteriorSLDS(discrete=qz, continuous=block_tridiag_mvn_smoother(potentials))


def update_discrete_posterior(model, params, qx: ContinuousPosteriorSLDS, x_samples: Array,
                              emissions: Array, inputs: Array) -> DiscretePosteriorSLDS:
    """Update the discrete posterior with the continuous posterior held fixed.

    Args:
        model: the SLDS
        params: model parameters
        qx: the continuous posterior
        x_samples: samples of the continuous states from ``qx``, shape (S, T, D)
        emissions: sequence of emissions
        inputs: sequence of corresponding inputs

    Returns:
        DiscretePosteriorSLDS: the updated discrete posterior.
    """
    transitions, dynamics, emission_component = model._components
    initial_probs = params.initial.probs
    log_likelihoods = dynamics.expected_log_likelihoods(params.dynamics, qx, inputs)
    if not emission_component.single_subspace:
        log_likelihoods = log_likelihoods + emission_component.expected_log_likelihoods(params.emissions, emissions, qx, inputs)
    if transitions.depends_on_x:
        expected_log_P = vmap(lambda x: transitions.log_transition_matrices(params.transitions, x, inputs))(x_samples).mean(0)
        log_c = logsumexp(expected_log_P, axis=-1)
        log_transition_matrices = expected_log_P - log_c[..., None]
        log_likelihoods = log_likelihoods.at[:-1].add(log_c)
    else:
        log_transition_matrices = transitions.log_transition_matrices(params.transitions, qx.smoothed_means, inputs)
    return _discrete_posterior(initial_probs, log_transition_matrices, log_likelihoods)


def backtracking_line_search(f: Callable, x: Array, dx: Array, g: Array,
                             alpha: float = 0.2, beta: float = 0.7, min_step: float = 1e-8) -> Scalar:
    """Find a step size along a search direction by backtracking.

    Starting from a step of 1, the step is multiplied by ``beta`` until the objective has decreased by
    at least ``alpha`` times the decrease predicted by the gradient. If no step above ``min_step``
    qualifies, the first step at or below ``min_step`` is returned.

    Args:
        f: objective to minimize
        x: current point
        dx: search direction
        g: gradient of ``f`` at ``x``
        alpha: fraction of the predicted decrease that the step must achieve
        beta: factor by which the step shrinks at each try
        min_step: the search stops at the first step at or below this value

    Returns:
        the step size
    """
    f0 = f(x)
    slope = jnp.sum(g * dx)

    def _sufficient_decrease(step):
        return f(x + step * dx) <= f0 + alpha * step * slope

    # The decision is carried instead of recomputed in the loop condition: under vmap, JAX evaluates the
    # condition twice, and in float32 the two results can disagree, so the loop would never end.
    def _body(carry):
        step, _ = carry
        step = beta * step
        return step, _sufficient_decrease(step)

    step, _ = lax.while_loop(lambda c: (c[0] > min_step) & ~c[1], _body,
                             (jnp.ones(()), _sufficient_decrease(jnp.ones(()))))
    return step


def newton_block_tridiag(f: Callable, grad_f: Callable, hess_blocks: Callable, x0: Array,
                         tolerance: float, maxiter: int) -> Array:
    """Minimize a function with Newton's method, using its block-tridiagonal Hessian.

    The Hessian of ``f`` must be block tridiagonal, with one (D, D) block per row of ``x``, and positive definite.

    Each step size is chosen by backtracking line search. Convergence is checked before each step,
    from the Newton decrement at the current point, so a point that has already converged is returned
    unchanged.

    Args:
        f: objective to minimize
        grad_f: gradient of ``f``
        hess_blocks: function returning the diagonal blocks and the blocks just below the diagonal of the
            Hessian of ``f``
        x0: starting point, shape (T, D)
        tolerance: stop when half the squared Newton decrement falls to or below this value
        maxiter: maximum number of Newton steps

    Returns:
        the minimizer, shape (T, D)
    """
    def _body(carry):
        x, _, it = carry
        J_diag, J_lower = hess_blocks(x)
        g = grad_f(x)
        dx = -block_tridiag_mvn_solve(J_diag, J_lower, g)
        lambdasq = -jnp.sum(g * dx)
        step = lax.cond(lambdasq / 2 <= tolerance,
                        lambda: jnp.zeros(()),
                        lambda: backtracking_line_search(f, x, dx, g))
        return x + step * dx, lambdasq, it + 1

    x, _, _ = lax.while_loop(lambda c: (c[1] / 2 > tolerance) & (c[2] < maxiter), _body, (x0, jnp.inf, 0))
    return x


def update_continuous_posterior(model, params, qz: DiscretePosteriorSLDS, qx: ContinuousPosteriorSLDS,
                                emissions: Array, inputs: Array) -> ContinuousPosteriorSLDS:
    """Update the continuous posterior with the discrete posterior held fixed.

    The new posterior is a Gaussian centered at the continuous states that maximize the log joint
    probability averaged over the discrete posterior, with precision equal to the negative Hessian there.
    It is exact when the transitions do not depend on the continuous state (one linear solve), and an
    approximation found by Newton's method when they do. For Newton's method the objective is divided by
    the number of time steps times the state dimension, so that the tolerance means the same at any size.

    Args:
        model: the SLDS
        params: model parameters
        qz: the discrete posterior
        qx: the current continuous posterior
        emissions: sequence of emissions
        inputs: sequence of corresponding inputs

    Returns:
        ContinuousPosteriorSLDS: the updated continuous posterior.
    """
    transitions, dynamics, emission_component = model._components
    num_timesteps, state_dim = qx.smoothed_means.shape
    scale = num_timesteps * state_dim

    def neg_expected_log_joint(x):
        lp = jnp.sum(xlogy(qz.smoothed_probs[0], params.initial.probs))
        lp += transitions.expected_log_trans_prob(params.transitions, x, inputs, qz.trans_probs)
        lp += dynamics.expected_log_joint_x(params.dynamics, x, qz.smoothed_probs, inputs)
        lp += emission_component.expected_log_joint_x(params.emissions, emissions, x, qz.smoothed_probs, inputs)
        return -lp

    def hess_blocks(x):
        J_diag, J_lower = dynamics.neg_hessian_blocks(params.dynamics, qz.smoothed_probs, inputs)
        J_diag = J_diag + emission_component.neg_hessian_blocks(params.emissions, emissions, x, qz.smoothed_probs, inputs)
        if transitions.depends_on_x:
            J_diag = J_diag.at[:-1].add(transitions.neg_hessian_x(params.transitions, x, inputs, qz.trans_probs))
        return J_diag, J_lower

    if not transitions.depends_on_x:
        x0 = qx.smoothed_means
        J_diag, J_lower = hess_blocks(x0)
        h = block_tridiag_mvn_matvec(J_diag, J_lower, x0) - jax.grad(neg_expected_log_joint)(x0)
        return block_tridiag_mvn_smoother(BlockTridiagPotentials(J_diag, J_lower, h))
    f = lambda x: neg_expected_log_joint(x) / scale
    x_star = newton_block_tridiag(f, jax.grad(f), lambda x: tree_map(lambda a: a / scale, hess_blocks(x)),
                                  qx.smoothed_means, model.newton_tolerance, model.newton_maxiter)
    J_diag, J_lower = hess_blocks(x_star)
    h = block_tridiag_mvn_matvec(J_diag, J_lower, x_star)
    return block_tridiag_mvn_smoother(BlockTridiagPotentials(J_diag, J_lower, h))


def discrete_entropy(qz: DiscretePosteriorSLDS) -> Scalar:
    """Compute the entropy of the discrete posterior.

    Pairs of states with zero posterior probability contribute zero.

    Args:
        qz: the discrete posterior

    Returns:
        the entropy
    """
    return (qz.log_normalizer
            - jnp.sum(xlogy(qz.smoothed_probs[0], qz.initial_probs))
            - jnp.sum(qz.smoothed_probs * qz.log_likelihoods)
            - jnp.sum(jnp.where(qz.trans_probs > 0, qz.trans_probs * qz.log_transition_matrices, 0.0)))


def slds_elbo(model, params, posterior: PosteriorSLDS, emissions: Array, inputs: Array, x_samples: Array) -> Scalar:
    """Compute the evidence lower bound (ELBO) on the log likelihood of one sequence.

    The ELBO is E_q[log p(z, x, y)] + entropy of q(z) + entropy of q(x). It is computed in closed form
    unless the transitions depend on the continuous state. In that case, the transitions' part of the
    first term, sum_t E_q[log p(z_{t+1} | z_t, x_t)], is averaged over ``x_samples``.

    Args:
        model: the SLDS
        params: model parameters
        posterior: approximate posterior
        emissions: sequence of emissions
        inputs: sequence of corresponding inputs
        x_samples: samples of the continuous states from the continuous posterior, shape (S, T, D)

    Returns:
        the ELBO
    """
    transitions, dynamics, emission_component = model._components
    qz, qx = posterior.discrete, posterior.continuous
    num_timesteps, state_dim = qx.smoothed_means.shape
    elbo = jnp.sum(xlogy(qz.smoothed_probs[0], params.initial.probs))
    if transitions.depends_on_x:
        log_P = vmap(lambda x: transitions.log_transition_matrices(params.transitions, x, inputs))(x_samples).mean(0)
        elbo += jnp.sum(qz.trans_probs * log_P)
    else:
        elbo += transitions.expected_log_trans_prob(params.transitions, qx.smoothed_means, inputs, qz.trans_probs)
    elbo += jnp.sum(qz.smoothed_probs * dynamics.expected_log_likelihoods(params.dynamics, qx, inputs))
    emission_terms = emission_component.expected_log_likelihoods(params.emissions, emissions, qx, inputs)
    elbo += jnp.sum(emission_component.state_weights(qz.smoothed_probs) * emission_terms)
    elbo += discrete_entropy(qz) + block_tridiag_mvn_entropy(qx.log_det_precision, num_timesteps, state_dim)
    return elbo
