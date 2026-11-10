"""Run the benchmark: both notebook models x timesteps x (ssm, dynamax), in float64.

Usage: python run.py
Each dataset has its own grid (SIZES): the SLDS from the notebook's 1000 steps up to 100k, the NASCAR rSLDS from the
notebook's 10000 steps up to 100k. Appends one JSON line per run to results.jsonl (skipping runs already present).
Each run is a subprocess so the libraries never share a process.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_data import path  # noqa: E402

# dynamax[dev] + ssm eb6c8aa, see dynamax/slds/fixtures/README.md
PYTHON = "/Users/nicholasbranigan/opt/miniconda3/envs/dyna_dev_ssm/bin/python"
RESULTS = os.path.join(HERE, "results.jsonl")
NUM_ITERS = 100
SIZES = {"slds": [1000, 10000, 100000], "rslds": [10000, 100000]}


def done(library, name, num_timesteps):
    if not os.path.exists(RESULTS):
        return False
    return any((r["library"], r["dataset"], r["num_timesteps"]) == (library, name, num_timesteps)
               for r in map(json.loads, open(RESULTS)))


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
    if out.returncode != 0 or not lines:
        print("FAILED:", " ".join(cmd), "\n", out.stderr[-2000:], file=sys.stderr, flush=True)
        return None
    return json.loads(lines[-1])


if __name__ == "__main__":
    grid = sorted((T, name) for name in SIZES for T in SIZES[name])   # cheap runs first
    for T, name in grid:
        if not os.path.exists(path(name, T)):
            subprocess.run([PYTHON, os.path.join(HERE, "make_data.py"), name, str(T)], check=True)   # sampled by ssm
    run([PYTHON, os.path.join(HERE, "bench_ssm.py"), grid[0][1], str(grid[0][0]), "2"])   # fill numba's on-disk cache
    for T, name in grid:
        for library in ("ssm", "dynamax"):
            if done(library, name, T):
                continue
            res = run([PYTHON, os.path.join(HERE, f"bench_{library}.py"), name, str(T), str(NUM_ITERS)])
            if res is None:
                continue
            with open(RESULTS, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"{library:8s} {name:6s} T={T:<8d} init {res['init_s']:7.2f}s  fit {res['fit_s']:8.2f}s  "
                  f"discrete state recovery {res['recovery']['discrete state recovery']:.3f}", flush=True)
