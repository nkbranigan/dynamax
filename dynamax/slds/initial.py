"""
Initial distribution of the discrete state of a switching linear dynamical system (SLDS).
"""
from typing import Union

import jax.numpy as jnp
from jaxtyping import Array

from dynamax.hidden_markov_model.models.initial import ParamsStandardHMMInitialState, StandardHMMInitialState
from dynamax.types import Scalar


class SLDSInitialState(StandardHMMInitialState):
    """Initial distribution of the discrete state, with a Dirichlet prior.

    :param num_states: number of discrete states
    :param initial_probs_concentration: concentration of the Dirichlet prior
    """

    def __init__(self, num_states: int, initial_probs_concentration: Union[Scalar, Array] = 1.0):
        super().__init__(num_states, initial_probs_concentration)

    def log_prior(self, params: ParamsStandardHMMInitialState) -> Scalar:
        """Compute the log prior probability of the parameters, or zero for a single state, where the
        Dirichlet is undefined."""
        if self.num_states == 1:
            return 0.0
        return super().log_prior(params)

    def m_step(self, params, props, batch_stats, m_step_state):
        """Update the initial probabilities to the mode of their posterior given the expected counts.

        The counts plus the prior are clipped at zero, and a small floor relative to the total is added so that no
        probability is exactly zero.
        """
        if props.probs.trainable:
            counts = jnp.maximum(batch_stats.sum(axis=0) + (self.initial_probs_concentration - 1.0), 0.0)
            counts = counts + 1e-8 * jnp.maximum(counts.sum(), 1.0)
            params = params._replace(probs=counts / counts.sum())
        return params, m_step_state
