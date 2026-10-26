"""
Tests for recovery against ``lindermanlab/ssm`` (commit eb6c8aa) on the datasets of ssm's two SLDS notebooks.
dynamax's fit and ssm's saved fit (``fixtures/ssm_fixtures.npz``) are scored on how well they recover the discrete
states, the continuous states, and the parameters. Each dynamax score must be at least ssm's minus 1% of its
magnitude. On the 1,000-step dataset, the solution dynamax reaches depends on the seed.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from dynamax.slds import GaussianSLDS, RecurrentGaussianSLDS
from dynamax.utils.utils import find_permutation

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ssm_fixtures.npz")
MODELS = {"slds": lambda: GaussianSLDS(5, 2, 10),
          "rslds": lambda: RecurrentGaussianSLDS(4, 2, 10, recurrent_only=True, dynamics_cov_type="diag")}
GROUPS = {"slds": ["P", "F", "b", "Q", "H", "d", "R"],
          "rslds": ["L", "W", "F", "b", "Q", "H", "d", "R"]}


def natural_params(params):
    """dynamax parameters named by the letters of the model docstrings, as ssm's saved ``recovery/<model>/true/`` and
    ``ssm/`` entries are."""
    Q = params.dynamics.covs
    out = {"F": params.dynamics.weights, "b": params.dynamics.biases, "Q": Q if Q.ndim == 3 else jax.vmap(jnp.diag)(Q),
           "H": params.emissions.weights[0], "d": params.emissions.biases[0], "R": params.emissions.covs[0]}
    if hasattr(params.transitions, "transition_matrix"):
        out["P"] = params.transitions.transition_matrix
    else:
        out["L"], out["W"] = params.transitions.logits[0], params.transitions.recurrent_weights
    return {k: np.asarray(v) for k, v in out.items()}


def in_true_coordinates(p, perm, coef):
    """Fitted parameters relabeled (fitted state k is true state perm[k]) and mapped by x = G x_hat + g, the
    least-squares fit of x on [x_hat, 1] with coefficients ``coef``."""
    inv = np.argsort(perm)                                  # true state j came from fitted state inv[j]
    G, g = coef[:-1].T, coef[-1]
    G_inv = np.linalg.inv(G)
    F = G @ p["F"][inv] @ G_inv
    H = p["H"] @ G_inv
    out = {"F": F, "b": p["b"][inv] @ G.T + g - F @ g, "Q": G @ p["Q"][inv] @ G.T, "H": H, "d": p["d"] - H @ g,
           "R": p["R"]}
    if "P" in p:
        out["P"] = p["P"][inv][:, inv]
    else:
        W = p["W"][inv] @ G_inv
        out["W"], out["L"] = W, p["L"][inv] - W @ g
    return out


def variance_explained(fitted, true, group):
    if group in ("L", "W"):                                 # softmax logits: remove the common shift over states
        fitted, true = fitted - fitted.mean(0), true - true.mean(0)
    return 1.0 - np.sum(np.abs(fitted - true) ** 2) / np.sum(np.abs(true) ** 2)


def recovery(name, p, x_hat, z_hat, data):
    """The recovery scores of one fit.

    Discrete state recovery is the share of time steps whose most likely state is the true one, after the best
    relabeling of the states. Continuous state recovery is the R^2 of the least-squares fit of the true continuous
    states on the posterior means. Each parameter's score is its variance explained, after relabeling the states and
    mapping the continuous states by that fit.

    Two more scores need no mapping of the continuous states: the eigenvalues of each state's F, and the subspace
    spanned by the columns of H (the smallest cosine between it and the true subspace). For the rSLDS, the regions
    score is the share of a 40 x 40 grid over the range of the true continuous states where the fitted and the true
    transitions pick the same most likely next state."""
    x, z = data[f"recovery/{name}/x"], data[f"recovery/{name}/z"]
    true = {k: data[f"recovery/{name}/true/{k}"] for k in GROUPS[name]}
    perm = np.asarray(find_permutation(z_hat, z))
    discrete = np.mean(perm[z_hat] == z)
    design = np.column_stack([x_hat, np.ones(len(x_hat))])
    coef = np.linalg.lstsq(design, x, rcond=None)[0]
    continuous = 1.0 - np.sum((design @ coef - x) ** 2) / np.sum((x - x.mean(0)) ** 2)
    aligned = in_true_coordinates(p, perm, coef)
    scores = {"discrete state recovery": float(discrete), "continuous state recovery": float(continuous)}
    scores.update({k: variance_explained(aligned[k], true[k], k) for k in GROUPS[name]})
    eigenvalues = lambda A: np.sort_complex(np.linalg.eigvals(A))      # sorted within each state
    scores["eigenvalues of F"] = variance_explained(eigenvalues(p["F"][np.argsort(perm)]), eigenvalues(true["F"]), "F")
    scores["subspace of H"] = np.linalg.svd(np.linalg.qr(p["H"])[0].T @ np.linalg.qr(true["H"])[0], compute_uv=False).min()
    if name == "rslds":
        grid = np.stack(np.meshgrid(*np.linspace(x.min(0), x.max(0), 40).T), -1).reshape(-1, x.shape[1])
        region = lambda W, L: np.argmax(grid @ W.T + L, axis=1)
        scores["regions of the plane"] = np.mean(region(aligned["W"], aligned["L"]) == region(true["W"], true["L"]))
    return scores


@pytest.mark.parametrize("name", list(MODELS))
def test_recovery_is_at_least_as_good_as_ssm(name):
    data = np.load(FIXTURE)
    y = jnp.asarray(data[f"recovery/{name}/y"])
    model = MODELS[name]()
    params, props = model.initialize(jr.PRNGKey(4), method="pca_arhmm", emissions=y)
    fitted, _, posterior = model.fit_em(params, props, y, verbose=False, return_posterior=True)
    posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)   # one sequence: drop the batch axis
    x_hat = np.asarray(posterior.continuous.smoothed_means)
    z_hat = np.asarray(model.most_likely_states(fitted, y, posterior))
    ours = recovery(name, natural_params(fitted), x_hat, z_hat, data)
    theirs = recovery(name, {k: data[f"recovery/{name}/ssm/{k}"] for k in GROUPS[name]}, data[f"recovery/{name}/x_hat"],
                      data[f"recovery/{name}/z_hat"], data)
    print(f"\n{name}")
    for k in ours:
        print(f"  {k:>26s}: dynamax {ours[k]:.4f} vs ssm {theirs[k]:.4f}")
    for k in ours:
        assert ours[k] >= theirs[k] - 0.01 * abs(theirs[k]), k
