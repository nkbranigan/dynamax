"""The large grid: a base point (N=100, D=10, K=10, T=10k, one sequence) and one axis varied at a time, for the
SLDS and the rSLDS, each fitted by dynamax float64 and ssm for 100 EM iterations.

Usage: python run_big.py [--iters 100] [--libs dynamax,ssm] [--only-dynamax-1m]
Appends to results_big.jsonl, skipping runs already present; cheaper runs first so partial results are useful.
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
SSM_PY = os.path.join(HERE, "ssm_venv", "bin", "python")
RESULTS = os.path.join(HERE, "results_big.jsonl")

BASE = dict(N=100, D=10, K=10, T=10_000, S=1)
VARIANTS = [dict(), dict(N=10), dict(D=2), dict(K=2), dict(K=5), dict(T=1_000), dict(D=20), dict(K=20),
            dict(T=1_000, S=10), dict(N=1_000), dict(T=100_000), dict(N=1_000, T=100_000)]
# ssm is skipped at N=1000, T=1M (days); dynamax tries it last.
TAIL = [dict(N=1_000, T=1_000_000)]


def key_of(r):
    return (r["library"], r["model"], r["N"], r["D"], r["K"], r["T"], r["S"], r["num_iters"])


def done():
    if not os.path.exists(RESULTS):
        return set()
    return {key_of(json.loads(l)) for l in open(RESULTS)}


def run(cmd, env=None):
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode != 0 or not lines:
        print("FAILED:", " ".join(cmd), "\n", out.stderr[-3000:], file=sys.stderr, flush=True)
        return None
    return json.loads(lines[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--libs", default="dynamax,ssm")
    args = ap.parse_args()
    libs = args.libs.split(",")
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    configs = [dict(BASE, **v) for v in VARIANTS] + [dict(BASE, **v) for v in TAIL]
    for cfg in configs:
        for model in ("slds", "rslds"):
            N, D, K, T, S = cfg["N"], cfg["D"], cfg["K"], cfg["T"], cfg["S"]
            data = os.path.join(REPO, "scratch_data", f"big_{model}_N{N}_D{D}_K{K}_T{T}_S{S}.npz")
            if not os.path.exists(data):
                gen = subprocess.run([DYNA_PY, os.path.join(HERE, "make_big_data.py"), model, str(N), str(D), str(K), str(T), str(S)],
                                     capture_output=True, text=True)
                if gen.returncode != 0:
                    print("DATA FAILED:", model, cfg, gen.stderr[-1000:], file=sys.stderr, flush=True)
                    continue
            for lib in libs:
                if lib == "ssm" and T >= 1_000_000:
                    continue
                if (lib, model, N, D, K, T, S, args.iters) in done():
                    continue
                start = time.time()
                if lib == "ssm":
                    res = run([SSM_PY, os.path.join(HERE, "bench_big_ssm.py"), data, str(args.iters)])
                else:
                    res = run([DYNA_PY, os.path.join(HERE, "bench_big_dynamax.py"), data, str(args.iters)],
                              env=dict(os.environ, JAX_ENABLE_X64="1"))
                if res is None:
                    continue
                res["dynamax_commit"] = commit
                res["wall_s"] = time.time() - start
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"{lib:8s} {model:6s} N={N:<5d} D={D:<3d} K={K:<3d} T={T:<8d} S={S:<4d} "
                      f"init {res.get('init_warm_s', res.get('init_s')):8.2f}s  per-iter {res['per_iter_s']:9.4f}s  "
                      f"objective {res['final_objective']:.1f}  wall {res['wall_s'] / 60:.1f} min", flush=True)
