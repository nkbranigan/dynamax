"""Run the benchmark: both notebook models x timesteps x (ssm, dynamax), in float64.

Usage: python run.py
Each dataset has its own grid (SIZES): the SLDS from the notebook's 1000 steps up to 100k, the NASCAR rSLDS from the
notebook's 10000 steps up to 100k. Sizes above the notebook's are batches of independent sequences of the notebook's
length. Samples every dataset and writes results.jsonl afresh, one JSON line per run. Each run is a subprocess so the
libraries never share a process.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# dynamax[dev] + ssm eb6c8aa, see dynamax/slds/fixtures/README.md
PYTHON = "/Users/nicholasbranigan/opt/miniconda3/envs/dyna_dev_ssm/bin/python"
RESULTS = os.path.join(HERE, "results.jsonl")
NUM_ITERS = 100
SIZES = {"slds": [1000, 10000, 100000], "rslds": [10000, 100000]}


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
        subprocess.run([PYTHON, os.path.join(HERE, "make_data.py"), name, str(T)], check=True)   # sampled by ssm
    run([PYTHON, os.path.join(HERE, "bench_ssm.py"), grid[0][1], str(grid[0][0]), "2"])   # fill numba's on-disk cache
    open(RESULTS, "w").close()
    for T, name in grid:
        for library in ("ssm", "dynamax"):
            res = run([PYTHON, os.path.join(HERE, f"bench_{library}.py"), name, str(T), str(NUM_ITERS)])
            if res is None:
                continue
            with open(RESULTS, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"{library:8s} {name:6s} T={T:<8d} init {res['init_s']:7.2f}s  fit {res['fit_s']:8.2f}s  "
                  f"discrete state recovery {res['recovery']['discrete state recovery']:.3f}", flush=True)
