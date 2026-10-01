"""Small diagonal-noise edge cases against independent NumPy float64 filtering.

Every prior/process covariance is strictly positive definite, and every
observation variance is positive. These exercise supported inputs rather than
assuming singular priors or noiseless observations have become supported.
"""

import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from dynamax.linear_gaussian_ssm.inference import lgssm_filter
from test_structured_filter import (
    _assert_matches_oracle,
    _dense_kalman_oracle,
    _make_case,
)


def _edge_case(name, dtype):
    k, d, t = 3, 9, 4
    params, observations, inputs = _make_case(
        k, d, t, dtype=dtype, time_varying=False,
        diagonal_noise=True, with_inputs=True,
    )
    h = np.asarray(params.emissions.weights, dtype=np.float64).copy()
    r = np.asarray(params.emissions.cov, dtype=np.float64).copy()
    if name == "zero_h":
        h[:] = 0.0
    elif name == "rank_deficient_h":
        # Exact duplicate columns leave a nontrivial latent direction unseen.
        h[:, 2] = h[:, 0]
        assert np.linalg.matrix_rank(h) == k - 1
    elif name == "unequal_positive_r":
        # Four orders of magnitude, with no zero or negative variance.
        r = np.geomspace(0.002, 20.0, d)
    elif name == "wide_multistep_strong_signal":
        # Full column rank, but many more observations than latent variables.
        # Include a substantial residual orthogonal to col(H), so scoring must
        # account for both the explained signal and unexplainable observation.
        h *= 128.0
        projection_basis, _ = np.linalg.qr(h)
        direction = np.linspace(-1.0, 1.0, d)
        orthogonal = direction - projection_basis @ (projection_basis.T @ direction)
        orthogonal /= np.linalg.norm(orthogonal)
        latent_targets = np.array([
            [20.0, -12.0, 8.0],
            [-9.0, 15.0, 6.0],
            [11.0, 7.0, -14.0],
            [4.0, -16.0, 12.0],
        ])
        observations = (
            latent_targets @ h.T
            + np.asarray(inputs) @ np.asarray(params.emissions.input_weights).T
            + np.asarray(params.emissions.bias)
            + np.array([12.0, -9.0, 15.0, -11.0])[:, None] * orthogonal
        )
        assert d > k and t > 1
        assert np.linalg.norm(h.T @ orthogonal) < 1e-10
    else:
        raise AssertionError(name)

    params = params._replace(emissions=params.emissions._replace(
        weights=jnp.asarray(h, dtype=dtype), cov=jnp.asarray(r, dtype=dtype),
    ))
    assert np.linalg.eigvalsh(np.asarray(params.initial.cov)).min() > 0.0
    assert np.linalg.eigvalsh(np.asarray(params.dynamics.cov)).min() > 0.0
    assert np.min(np.asarray(params.emissions.cov)) > 0.0
    return params, jnp.asarray(observations, dtype=dtype), inputs


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("name", [
    "zero_h", "rank_deficient_h", "unequal_positive_r", "wide_multistep_strong_signal",
])
def test_diagonal_filter_edge_cases_match_dense_float64_oracle(name, dtype):
    """Check likelihood, every posterior moment, and retained positive variance."""
    with enable_x64(dtype == np.float64):
        params, observations, inputs = _edge_case(name, dtype)
        expected = _dense_kalman_oracle(
            params, observations, inputs, diagonal_noise=True,
        )
        posterior = lgssm_filter(params, observations, inputs)
        tolerance = 8e-5 if dtype == np.float32 else 3e-9
        _assert_matches_oracle(posterior, expected, dtype, tolerance=tolerance)

        # An absolute allclose tolerance could let an incorrectly zeroed tiny
        # posterior covariance pass. Check its relative matrix error as well.
        actual_cov = np.asarray(posterior.filtered_covariances, dtype=np.float64)
        expected_cov = expected[2]
        errors = np.linalg.norm(actual_cov - expected_cov, axis=(-2, -1))
        scales = np.linalg.norm(expected_cov, axis=(-2, -1))
        assert np.all(errors / scales < tolerance)
        assert np.linalg.eigvalsh(actual_cov).min() > 0.0

        if name == "zero_h":
            # No state information is present: every filtered moment must be
            # the unconditioned state prediction, despite nonzero observations.
            mean = np.asarray(params.initial.mean, dtype=np.float64)
            cov = np.asarray(params.initial.cov, dtype=np.float64)
            f = np.asarray(params.dynamics.weights, dtype=np.float64)
            q = np.asarray(params.dynamics.cov, dtype=np.float64)
            b = np.asarray(params.dynamics.bias, dtype=np.float64)
            u_weights = np.asarray(params.dynamics.input_weights, dtype=np.float64)
            for t in range(observations.shape[0]):
                np.testing.assert_allclose(
                    posterior.filtered_means[t], mean, rtol=tolerance, atol=tolerance,
                )
                np.testing.assert_allclose(
                    actual_cov[t], cov, rtol=tolerance, atol=tolerance,
                )
                mean = f @ mean + b + u_weights @ np.asarray(inputs[t])
                cov = f @ cov @ f.T + q
