#!/usr/bin/env python3
"""Reproducible, matched outer-JIT diagonal-noise LGSSM filter benchmark.

Baseline is frozen at runtime from the specified git ref (default 5098927),
or read from an explicit baseline file. The candidate imports the worktree.
Run on an otherwise idle machine; this script does not change repository files.
"""
import os
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_compilation_cache", False)
jax.config.update("jax_enable_x64", False)

DEFAULT_BASELINE_REF = "5098927f2c377c0dd548cda750e6f1436220aac6"


def default_worktree():
    """Find the repository containing this script, or use the current directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "dynamax/linear_gaussian_ssm/inference.py").is_file() and (parent / "pyproject.toml").is_file():
            return str(parent)
    return str(Path.cwd())


def ready(value):
    return jax.block_until_ready(value)


def command(command_args):
    try:
        return subprocess.check_output(command_args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as error:
        return f"unavailable: {type(error).__name__}"


def parse_case(text):
    try:
        value = tuple(int(part) for part in re.split("[,x:]", text))
        assert len(value) == 3 and all(x > 0 for x in value)
        return value
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError("Case must be T,K,D, for example 32,8,1024")


def make_data(length, state_dim, emission_dim, seed):
    """All setup uses host NumPy; arrays are explicit runtime JIT arguments."""
    rng = np.random.default_rng(seed + length + 97 * state_dim + 1009 * emission_dim)
    k, d, p = state_dim, emission_dim, 3
    mean = rng.normal(scale=0.1, size=k)
    initial_cov = 1.2 * np.eye(k) + 0.05 * np.ones((k, k)) / k
    f = 0.9 * np.eye(k) + rng.normal(scale=0.015 / np.sqrt(k), size=(k, k))
    q = 0.15 * np.eye(k) + 0.01 * np.ones((k, k)) / k
    b = rng.normal(scale=0.05, size=k)
    b_inputs = rng.normal(scale=0.1, size=(k, p))
    h = rng.normal(scale=1 / np.sqrt(k), size=(d, k))
    r = np.exp(rng.uniform(-0.5, 0.5, size=d))
    bias = rng.normal(scale=0.1, size=d)
    input_weights = rng.normal(scale=0.08, size=(d, p))
    inputs = rng.normal(size=(length, p))
    observations = []
    state = mean + np.linalg.cholesky(initial_cov) @ rng.normal(size=k)
    q_chol = np.linalg.cholesky(q)
    for t in range(length):
        observations.append(h @ state + input_weights @ inputs[t] + bias + np.sqrt(r) * rng.normal(size=d))
        state = f @ state + b_inputs @ inputs[t] + b + q_chol @ rng.normal(size=k)
    return {name: np.asarray(value, dtype=np.float32) for name, value in dict(
        initial_mean=mean, initial_cov=initial_cov, dynamics_weights=f,
        dynamics_cov=q, dynamics_bias=b, dynamics_input_weights=b_inputs,
        emissions_weights=h, emissions_cov=r, emissions_bias=bias,
        emissions_input_weights=input_weights, inputs=inputs,
        observations=np.stack(observations)).items()}


def params_for(module, data):
    return module.make_lgssm_params(**{name: jax.device_put(value)
                                      for name, value in data.items()
                                      if name not in ("inputs", "observations")})


def memory_record(compiled):
    try:
        analysis = compiled.memory_analysis()
        if analysis is None:
            return {"available": False}
        result = {"available": True, "repr": str(analysis)}
        for name in ["argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                     "temp_size_in_bytes", "generated_code_size_in_bytes",
                     "host_argument_size_in_bytes", "host_output_size_in_bytes",
                     "host_alias_size_in_bytes", "host_temp_size_in_bytes"]:
            value = getattr(analysis, name, None)
            if value is not None:
                result[name] = int(value)
        return result
    except Exception as error:
        return {"available": False, "error": repr(error)}


def inspect_hlo(compiled, case_id, variant, length, state_dim, emission_dim, output_dir):
    text = compiled.as_text()
    path = output_dir / f"{case_id}_{variant}.optimized_hlo.txt"
    path.write_text(text)
    # Optimized HLO uses shapes such as f32[1024,1024]{1,0}; inspect all lines.
    matches = []
    for number, line in enumerate(text.splitlines(), 1):
        for dimensions in re.findall(r"(?:bf16|f16|f32|f64|c64|c128|s32|s64|u32|u64|pred)\[([0-9,]+)\]", line):
            shape = tuple(map(int, dimensions.split(",")))
            if any(shape[index:index + 2] == (emission_dim, emission_dim) for index in range(len(shape) - 1)):
                matches.append({"line": number, "shape": list(shape), "text": line[:600]})
                break
    # T == D would make the observations legitimately D x D; do not mislabel it.
    collision = emission_dim in (length, state_dim, 3)
    if variant == "candidate" and not collision:
        assert not matches, f"Candidate HLO has observation-space D x D shapes: {matches[:3]}"
    return {"path": str(path), "size_bytes": len(text.encode()),
            "observation_space_square_shape_match_count": len(matches),
            "shape_matches": matches[:30], "dimension_collision": collision,
            "candidate_no_D_by_D_assertion": "skipped_dimension_collision" if collision else ("passed" if variant == "candidate" else "not_applicable")}


def posterior_record(value):
    return {"filtered_means": np.asarray(value.filtered_means),
            "filtered_covariances": np.asarray(value.filtered_covariances),
            "marginal_loglik": np.asarray(value.marginal_loglik)}


def compare_posteriors(reference, candidate, atol, rtol, *, enforce=False,
                       loglik_atol=None, loglik_rtol=None):
    """Report errors against a reference; only the candidate/oracle check gates timing."""
    report = {}
    passed = True
    for name in ("filtered_means", "filtered_covariances", "marginal_loglik"):
        # Error measurement itself is float64, including baseline float32 values.
        a, b = np.asarray(reference[name], dtype=np.float64), np.asarray(candidate[name], dtype=np.float64)
        item_atol = loglik_atol if name == "marginal_loglik" and loglik_atol is not None else atol
        item_rtol = loglik_rtol if name == "marginal_loglik" and loglik_rtol is not None else rtol
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        close = bool(finite and np.allclose(b, a, atol=item_atol, rtol=item_rtol))
        passed = passed and close
        if enforce:
            assert finite, name
            np.testing.assert_allclose(b, a, atol=item_atol, rtol=item_rtol, err_msg=name)
        denominator = float(np.linalg.norm(a.ravel()))
        error_norm = float(np.linalg.norm((a - b).ravel())) if finite else None
        report[name] = {"max_abs_error": float(np.max(np.abs(a - b))) if finite else None,
                        "relative_frobenius_error": error_norm / max(denominator, np.finfo(np.float64).tiny) if finite else None,
                        "error_frobenius_norm": error_norm,
                        "reference_frobenius_norm": denominator,
                        "reference_max_abs": float(np.max(np.abs(a))),
                        "finite": finite, "passed": close, "atol": item_atol, "rtol": item_rtol}
    covariances = candidate["filtered_covariances"]
    min_eigenvalue = float(np.linalg.eigvalsh(covariances.astype(np.float64)).min()) if np.isfinite(covariances).all() else None
    if enforce:
        assert min_eigenvalue is not None and min_eigenvalue > 0, min_eigenvalue
    report.update(passed=passed, enforced=enforce, atol=atol, rtol=rtol,
                  candidate_min_filtered_covariance_eigenvalue=min_eigenvalue)
    return report


def numpy_information_reference(data):
    """Independent float64 information-form reference for every case.

    It avoids emission-space square matrices; it is not a timing comparator.
    """
    a = {name: value.astype(np.float64) for name, value in data.items()}
    mean, cov = a["initial_mean"], a["initial_cov"]
    h, r = a["emissions_weights"], a["emissions_cov"]
    means, covs, loglik = [], [], 0.0
    weighted_h = h / r[:, None]
    for u, y in zip(a["inputs"], a["observations"]):
        residual = y - h @ mean - a["emissions_input_weights"] @ u - a["emissions_bias"]
        precision = np.linalg.solve(cov, np.eye(len(mean))) + h.T @ weighted_h
        filtered_cov = np.linalg.solve(precision, np.eye(len(mean)))
        information = h.T @ (residual / r)
        filtered_mean = mean + filtered_cov @ information
        logdet = np.log(r).sum() + np.linalg.slogdet(cov)[1] + np.linalg.slogdet(precision)[1]
        quadratic = np.dot(residual, residual / r) - information @ filtered_cov @ information
        loglik += -0.5 * (len(y) * np.log(2 * np.pi) + logdet + quadratic)
        means.append(filtered_mean); covs.append(filtered_cov)
        mean = a["dynamics_weights"] @ filtered_mean + a["dynamics_input_weights"] @ u + a["dynamics_bias"]
        cov = a["dynamics_weights"] @ filtered_cov @ a["dynamics_weights"].T + a["dynamics_cov"]
    return {"filtered_means": np.stack(means), "filtered_covariances": np.stack(covs),
            "marginal_loglik": np.asarray(loglik)}


def data_fingerprint(data):
    """Hash exact float32 model, inputs and observations used by both filters."""
    digest = hashlib.sha256()
    for name in sorted(data):
        value = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", default=default_worktree())
    parser.add_argument("--baseline", help="Read this exact baseline inference.py instead of exporting a git ref")
    parser.add_argument("--ref", "--baseline-ref", dest="baseline_ref", default=DEFAULT_BASELINE_REF,
                        help="Commit/ref to freeze as the baseline when --baseline is omitted")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "filter_benchmark_results",
                        help="Directory for frozen baseline, HLO files and default JSON output")
    parser.add_argument("--output", type=Path, help="Override JSON output path; other artifacts still use --output-dir")
    parser.add_argument("--case", type=parse_case, action="append", help="Matched T,K,D case; repeat to define custom grid")
    parser.add_argument("--candidate-only-case", type=parse_case, action="append", default=[], help="Additional capacity case, e.g.16,16,8192; no baseline compilation or execution")
    parser.add_argument("--warm-repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=5098927)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--oracle-moment-atol", type=float, default=1e-6)
    parser.add_argument("--oracle-moment-rtol", type=float, default=1e-4)
    parser.add_argument("--oracle-loglik-atol", type=float, default=1e-3)
    parser.add_argument("--oracle-loglik-rtol", type=float, default=5e-5)
    args = parser.parse_args()
    assert args.warm_repeats >= 15, "At least 15 warm measurements per variant"
    worktree = Path(args.worktree).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output.resolve() if args.output is not None else output_dir / "filter_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline is not None:
        baseline_path = Path(args.baseline).resolve()
        baseline_commit = None
        baseline_provenance = "explicit baseline file; commit identity not inferred"
    else:
        baseline_commit = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"], text=True).strip()
        baseline_bytes = subprocess.check_output(
            ["git", "-C", str(worktree), "show", f"{baseline_commit}:dynamax/linear_gaussian_ssm/inference.py"])
        baseline_path = output_dir / "baseline_inference.py"
        baseline_path.write_bytes(baseline_bytes)
        (output_dir / "baseline_commit.txt").write_text(baseline_commit + "\n")
        baseline_provenance = "git show of fixed baseline ref"
    sys.path.insert(0, str(worktree))
    from dynamax.linear_gaussian_ssm import inference as candidate
    assert Path(candidate.__file__).resolve().is_relative_to(worktree), candidate.__file__
    spec = importlib.util.spec_from_file_location("audit_frozen_lgssm_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    variants = {"baseline": baseline, "candidate": candidate}
    cases = [(case, False) for case in (args.case or [(32, 8, 64), (32, 8, 256), (32, 8, 1024), (32, 16, 2048)])]
    cases += [(case, True) for case in args.candidate_only_case]
    rng = random.Random(args.seed)
    output = {"metadata": {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "logical_cpu_count": os.cpu_count(),
        "hardware": {name: command(["/usr/sbin/sysctl", "-n", name]) for name in ["hw.model", "machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu"]},
        "versions": {name: metadata.version(name) for name in ["jax", "jaxlib", "tfp-nightly", "numpy"]},
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "thread_environment": {name: os.environ.get(name) for name in ["XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "JAX_PLATFORMS"]},
        "dtype": "float32", "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_enable_compilation_cache": False,
        "baseline_source": str(baseline_path),
        "baseline_commit": baseline_commit,
        "baseline_ref_requested": args.baseline_ref if args.baseline is None else None,
        "baseline_provenance": baseline_provenance,
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "candidate_source": candidate.__file__,
        "candidate_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        "candidate_head": command(["git", "-C", str(worktree), "rev-parse", "HEAD"]),
        "candidate_git_diff": command(["git", "-C", str(worktree), "diff", "--", "dynamax/linear_gaussian_ssm/inference.py"]),
        "warm_repeats": args.warm_repeats, "seed": args.seed,
        "data_seed_formula": "seed + T + 97*K + 1009*D; unchanged NumPy default_rng model/data generation",
        "output_directory": str(output_dir),
        "comparison": "Both outer-jitted; all parameter, emission and input arrays are dynamic runtime arguments; no precomputed factors",
        "compile_timing": "clear in-memory caches per variant; lower and compile measured separately; imports/backend setup excluded",
        "warm_timing": "Validation precedes timings; alternating variant order each round, synchronized with block_until_ready; 3 extra warmup calls per variant",
        "memory_note": "XLA compiled buffer estimates, not process peak RSS; optimized HLO square-shape scan is structural evidence, not a formal allocation proof",
        "model": "Static dense H,F,Q,P; positive diagonal R vector; nonzero biases and 3-dimensional inputs; same deterministic data for both variants",
        "correctness": "Every case uses an independent NumPy float64 information-form oracle on the identical float32 inputs; candidate moment/loglik tolerances gate warm timings; baseline errors and prior comparison tolerances are diagnostic only",
    }, "results": []}
    started_all = time.perf_counter()
    for (length, state_dim, emission_dim), candidate_only in cases:
        case_id = f"T{length}_K{state_dim}_D{emission_dim}"
        data = make_data(length, state_dim, emission_dim, args.seed)
        shared_observations, shared_inputs = jax.device_put(data["observations"]), jax.device_put(data["inputs"])
        selected = ["candidate"] if candidate_only else ["baseline", "candidate"]
        build_order = list(selected); rng.shuffle(build_order)
        executables, call_args, initial_outputs = {}, {}, {}
        record = {"id": case_id, "T": length, "K": state_dim, "D": emission_dim,
                  "candidate_only": candidate_only, "compile_order": build_order, "variants": {},
                  "data_sha256": data_fingerprint(data),
                  "derived_numpy_seed": args.seed + length + 97 * state_dim + 1009 * emission_dim}
        for name in build_order:
            params = params_for(variants[name], data)
            call_args[name] = ready((params, shared_observations, shared_inputs))
            jax.clear_caches()
            fn = jax.jit(variants[name].lgssm_filter)
            before = time.perf_counter_ns()
            lowered = fn.lower(*call_args[name])
            lower_ms = (time.perf_counter_ns() - before) / 1e6
            before = time.perf_counter_ns()
            executable = lowered.compile()
            compile_ms = (time.perf_counter_ns() - before) / 1e6
            executables[name] = executable
            record["variants"][name] = {"lower_ms": lower_ms, "compile_ms": compile_ms,
                "lower_plus_compile_ms": lower_ms + compile_ms,
                "compiled_memory": memory_record(executable),
                "optimized_hlo": inspect_hlo(executable, case_id, name, length, state_dim, emission_dim, output_dir)}
            initial_outputs[name] = posterior_record(ready(executable(*call_args[name])))
        oracle = numpy_information_reference(data)
        record["oracle_accuracy"] = {name: compare_posteriors(
            oracle, initial_outputs[name], args.oracle_moment_atol, args.oracle_moment_rtol,
            enforce=(name == "candidate"), loglik_atol=args.oracle_loglik_atol,
            loglik_rtol=args.oracle_loglik_rtol) for name in selected}
        record["correctness"] = record["oracle_accuracy"]["candidate"].copy()
        record["correctness"]["reference"] = "independent NumPy float64 information-form oracle (untimed)"
        if not candidate_only:
            record["baseline_comparison"] = compare_posteriors(
                initial_outputs["baseline"], initial_outputs["candidate"], args.atol, args.rtol)
            record["baseline_comparison"]["reference"] = "frozen baseline; diagnostic only, not the correctness oracle"
        for _ in range(3):
            for name in selected:
                ready(executables[name](*call_args[name]))
        samples = {name: [] for name in selected}
        first_order = list(selected); rng.shuffle(first_order)
        record["warm_round_orders"] = []
        for repeat in range(args.warm_repeats):
            order = first_order if repeat % 2 == 0 else first_order[::-1]
            record["warm_round_orders"].append(order)
            for name in order:
                before = time.perf_counter_ns()
                ready(executables[name](*call_args[name]))
                samples[name].append((time.perf_counter_ns() - before) / 1e6)
        for name, times in samples.items():
            record["variants"][name].update(warm_ms=times, warm_median_ms=statistics.median(times),
                warm_p10_ms=float(np.percentile(times, 10)), warm_p90_ms=float(np.percentile(times, 90)))
        if not candidate_only:
            record["warm_median_speedup"] = record["variants"]["baseline"]["warm_median_ms"] / record["variants"]["candidate"]["warm_median_ms"]
        output["results"].append(record)
        output["elapsed_seconds"] = time.perf_counter() - started_all
        destination.write_text(json.dumps(output, indent=2) + "\n")
        print(case_id, {name: round(record["variants"][name]["warm_median_ms"], 4) for name in selected},
              "speedup", round(record.get("warm_median_speedup", 0), 2), flush=True)
    print(f"Wrote {destination}; elapsed {output['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
