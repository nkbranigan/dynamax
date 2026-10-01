"""Filtered-moment derivatives checked against finite differences of SciPy filtering."""

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from dynamax.linear_gaussian_ssm.inference import lgssm_filter
from test_structured_filter import _dense_kalman_oracle, _make_case


def test_filtered_moment_directional_derivatives_against_dense_oracle():
    """Exercise differentiation through moments, including multiple prediction steps."""
    with enable_x64(True):
        params, observations, inputs = _make_case(
            2, 5, 4, dtype=np.float64, time_varying=True,
            diagonal_noise=True, with_inputs=True,
        )
        arguments = (
            np.asarray(params.emissions.weights),
            np.log(np.asarray(params.emissions.cov)),
            np.linalg.cholesky(np.asarray(params.initial.cov)),
            np.asarray(params.dynamics.weights),
        )
        rng = np.random.default_rng(5831)
        mean_weights = rng.normal(size=(4, 2))
        covariance_weights = 3.0 * rng.normal(size=(4, 2, 2))
        covariance_weights = (
            covariance_weights + np.swapaxes(covariance_weights, -1, -2)
        ) / 2.0

        def with_arguments(args, xp):
            h, log_r, initial_chol, f = args
            factor = xp.tril(initial_chol)
            return params._replace(
                initial=params.initial._replace(cov=factor @ factor.T),
                dynamics=params.dynamics._replace(weights=f),
                emissions=params.emissions._replace(weights=h, cov=xp.exp(log_r)),
            )

        def candidate(*args):
            posterior = lgssm_filter(with_arguments(args, jnp), observations, inputs)
            return (
                jnp.sum(posterior.filtered_means * mean_weights)
                + jnp.sum(posterior.filtered_covariances * covariance_weights)
            )

        def reference(args):
            _, means, covariances = _dense_kalman_oracle(
                with_arguments(args, np), observations, inputs, diagonal_noise=True,
            )
            return np.sum(means * mean_weights) + np.sum(covariances * covariance_weights)

        value, gradients = jax.value_and_grad(candidate, argnums=(0, 1, 2, 3))(
            *(jnp.asarray(arg, dtype=jnp.float64) for arg in arguments)
        )
        assert value.dtype == jnp.float64
        np.testing.assert_allclose(value, reference(arguments), rtol=2e-10, atol=2e-10)

        names = ("H", "log positive R", "initial Cholesky factor", "dynamics weights")
        step = 1e-5
        for index, (name, argument, gradient) in enumerate(zip(names, arguments, gradients)):
            assert gradient.dtype == jnp.float64, name
            gradient = np.asarray(gradient)
            assert np.all(np.isfinite(gradient)), name
            assert np.linalg.norm(gradient) > 1e-7, f"trivial moment gradient for {name}"
            direction = rng.normal(size=argument.shape)
            if index == 2:
                direction = np.tril(direction)
            direction /= np.linalg.norm(direction)
            plus, minus = list(arguments), list(arguments)
            plus[index] = argument + step * direction
            minus[index] = argument - step * direction
            expected = (reference(plus) - reference(minus)) / (2.0 * step)
            actual = np.sum(gradient * direction)
            assert np.isfinite(expected), name
            np.testing.assert_allclose(
                actual, expected, rtol=2e-6, atol=2e-8,
                err_msg=f"filtered-moment directional derivative differs for {name}",
            )
