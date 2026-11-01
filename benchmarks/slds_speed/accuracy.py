"""Recovery measures shared by the dynamax and ssm bench scripts (numpy + scipy only, so both environments can
import it). Sequences may be batched: states (S, T) or (T,), latents (S, T, D) or (T, D); batches are pooled."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def state_accuracy(z_true, z_hat):
    """Fraction of timesteps whose inferred discrete state matches the true one after the best relabelling
    (a maximum-overlap assignment of inferred to true states, as ssm's find_permutation)."""
    z_true, z_hat = np.asarray(z_true).reshape(-1).astype(int), np.asarray(z_hat).reshape(-1).astype(int)
    num_states = max(z_true.max(), z_hat.max()) + 1
    overlap = np.zeros((num_states, num_states))
    np.add.at(overlap, (z_true, z_hat), 1)
    rows, cols = linear_sum_assignment(-overlap)
    return float(overlap[rows, cols].sum() / z_true.size)


def latent_r2(x_true, x_hat):
    """R^2 of the least-squares affine fit x_true ~ G x_hat + g, pooled over latent dimensions: the latents
    are identifiable only up to an invertible affine map, which the fit absorbs."""
    x_true, x_hat = np.asarray(x_true, dtype=np.float64), np.asarray(x_hat, dtype=np.float64)
    x_true, x_hat = x_true.reshape(-1, x_true.shape[-1]), x_hat.reshape(-1, x_hat.shape[-1])
    design = np.concatenate([x_hat, np.ones((x_hat.shape[0], 1))], axis=1)
    coef = np.linalg.lstsq(design, x_true, rcond=None)[0]
    residual = np.sum((x_true - design @ coef) ** 2)
    total = np.sum((x_true - x_true.mean(axis=0)) ** 2)
    return float(1.0 - residual / total)
