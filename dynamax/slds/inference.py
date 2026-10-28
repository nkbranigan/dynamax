"""
The tuples that hold the approximate posterior over the states of a switching linear dynamical system (SLDS).
"""
from typing import NamedTuple

from jaxtyping import Array, Float

from dynamax.linear_gaussian_ssm.block_tridiag import BlockTridiagPosterior
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
