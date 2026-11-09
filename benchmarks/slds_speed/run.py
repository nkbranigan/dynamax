"""Run the speed grid: both notebook models x timesteps x (ssm, dynamax), in float64.

Usage: python run.py [--iters 100] [--timesteps 1000,10000,100000] [--datasets slds,rslds] [--libs ssm,dynamax64]
Without --timesteps each dataset uses its own grid (SIZES): the SLDS from the notebook's 1000 steps up to 1M, the
NASCAR rSLDS from the notebook's 10000 steps up to 1M. dynamax32 is accepted as a library but not run by default.
Appends one JSON line per run to results.jsonl (skipping runs already present) and records the
dynamax commit measured. Each run is a subprocess so the libraries never share a process.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DYNA_PY = "/Users/nicholasbranigan/opt/miniconda3/envs/dyna_dev/bin/python"
SSM_PY = "/Users/nicholasbranigan/opt/miniconda3/envs/dyna_dev_ssm/bin/python"   # dynamax[dev] + ssm eb6c8aa, see dynamax/slds/fixtures/README.md
RESULTS = os.path.join(HERE, "results.jsonl")
SIZES = {"slds": [1000, 10000, 100000, 1000000], "rslds": [10000, 100000, 1000000]}


def done(key):
    if not os.path.exists(RESULTS):
        return False
    for line in open(RESULTS):
        r = json.loads(line)
        if (r.get("library"), r.get("dtype"), r.get("dataset"), r.get("num_timesteps"), r.get("num_iters")) == key:
            return True
    return False


def run(cmd, env=None):
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode != 0 or not lines:
        print("FAILED:", " ".join(cmd), "\n", out.stderr[-2000:], file=sys.stderr, flush=True)
        return None
    return json.loads(lines[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--timesteps", default=None)
    ap.add_argument("--datasets", default="slds,rslds")
    ap.add_argument("--libs", default="ssm,dynamax64")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    datasets = args.datasets.split(",")
    sizes = {name: ([int(t) for t in args.timesteps.split(",")] if args.timesteps else SIZES[name]) for name in datasets}
    for T, name in sorted((T, name) for name in datasets for T in sizes[name]):   # cheap runs first
        data = os.path.join(REPO, "scratch_data", f"{name}_T{T}.npz")
        if not os.path.exists(data):
            subprocess.run([SSM_PY, os.path.join(HERE, "make_data.py"), name, str(T)], check=True)   # sampled by ssm
        for lib in args.libs.split(","):
            dtype = "float64" if lib in ("ssm", "dynamax64") else "float32"
            key = ("ssm" if lib == "ssm" else "dynamax", dtype, name, T, args.iters)
            if done(key):
                continue
            start = time.time()
            if lib == "ssm":
                res = run([SSM_PY, os.path.join(HERE, "bench_ssm.py"), name, str(T), str(args.iters)])
            else:
                env = dict(os.environ, JAX_ENABLE_X64="1" if lib == "dynamax64" else "0")
                res = run([DYNA_PY, os.path.join(HERE, "bench_dynamax.py"), name, str(T), str(args.iters)], env=env)
            if res is None:
                continue
            res["dynamax_commit"] = commit
            res["wall_s"] = time.time() - start
            with open(RESULTS, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"{lib:9s} {name:8s} T={T:<8d} init {res['init_s']:7.2f}s  "
                  f"per-iter {res['per_iter_s']:8.4f}s  objective {res['final_objective']:.1f}", flush=True)
