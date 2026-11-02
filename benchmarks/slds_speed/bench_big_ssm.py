"""Time lindermanlab/ssm on a dataset written by make_big_data.py.

Usage: <python with lindermanlab/ssm installed> bench_big_ssm.py <npz> <num_iters>
Same protocol as bench_ssm.py: numba warmed up on a tiny model, then initialization and Laplace EM timed
once each. Batched sequences are passed as a list.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import numpy.random as npr

warnings.filterwarnings("ignore")
import ssm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy import latent_r2, state_accuracy  # noqa: E402


def build(model_name, N, K, D):
    transitions = "standard" if model_name == "slds" else "recurrent_only"
    return ssm.SLDS(N, K, D, transitions=transitions, dynamics="gaussian", emissions="gaussian", single_subspace=True)


def run(model_name, N, K, D, datas, num_iters, num_init_iters=50):
    npr.seed(0)
    model = build(model_name, N, K, D)
    start = time.perf_counter()
    model.initialize(datas, num_init_iters=num_init_iters)
    init_s = time.perf_counter() - start
    start = time.perf_counter()
    out = model.fit(datas, method="laplace_em", variational_posterior="structured_meanfield",
                    num_iters=num_iters, initialize=False, verbose=0)
    fit_s = time.perf_counter() - start
    elbos, q = out
    return init_s, fit_s, np.asarray(elbos), model, q


if __name__ == "__main__":
    npz, num_iters = sys.argv[1], int(sys.argv[2])
    d = np.load(npz)
    model_name, N, D, K, T, S = str(d["model"]), int(d["N"]), int(d["D"]), int(d["K"]), int(d["T"]), int(d["S"])
    datas = [np.ascontiguousarray(d["emissions"][s]) for s in range(S)]
    run(model_name, N, K, D, [datas[0][:200]], 2, num_init_iters=2)       # warm up numba
    init_s, fit_s, elbos, model, q = run(model_name, N, K, D, datas, num_iters)
    x_hat = np.stack(q.mean_continuous_states)
    z_hat = np.stack([model.most_likely_states(x, data) for x, data in zip(q.mean_continuous_states, datas)])
    print(json.dumps({
        "library": "ssm", "dtype": "float64", "model": model_name,
        "N": N, "D": D, "K": K, "T": T, "S": S, "num_iters": num_iters,
        "init_s": init_s, "fit_s": fit_s, "per_iter_s": fit_s / num_iters,
        "final_objective": float(elbos[-1]),
        "objective_monotone": bool(np.all(np.diff(elbos) >= -1e-3 * abs(elbos[-1]))),
        "state_accuracy": state_accuracy(d["states"], z_hat), "latent_r2": latent_r2(d["latents"], x_hat),
    }))
