#!/usr/bin/env python3
"""Attribute diagonal-filter gains: shared owned kernel versus original TFP likelihood.

Production files are read-only. This constructs a registered scratch module
which preserves optimized conditioning but replaces its consumed likelihood
with the exact frozen-baseline _log_likelihood function. The unused owned
likelihood is allowed to disappear through XLA dead-code elimination, as it
would in a real implementation. Consequently this measures the net compiled
effect of sharing/owning the likelihood, not Python package import overhead.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKTREE = next(
    (parent for parent in DEFAULT_ROOT.parents
     if (parent / "dynamax/linear_gaussian_ssm/inference.py").is_file()),
    Path.cwd(),
)
DEFAULT_HARNESS = DEFAULT_ROOT / "benchmark_structured_filter.py"
if not DEFAULT_HARNESS.is_file():
    DEFAULT_HARNESS = DEFAULT_ROOT / "benchmark_filter.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_ablation_source(candidate_source, baseline_source):
    baseline_tree = ast.parse(baseline_source)
    baseline_filter = next(node for node in baseline_tree.body
                           if isinstance(node, ast.FunctionDef) and node.name == "lgssm_filter")
    likelihood_node = next(node for node in baseline_filter.body
                           if isinstance(node, ast.FunctionDef) and node.name == "_log_likelihood")
    likelihood = textwrap.dedent("\n".join(baseline_source.splitlines()[likelihood_node.lineno - 1:likelihood_node.end_lineno]))
    likelihood = likelihood.replace("def _log_likelihood(", "def _audit_baseline_log_likelihood(", 1)
    candidate_tree = ast.parse(candidate_source)
    candidate_filter = next(node for node in candidate_tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == "lgssm_filter")
    original_filter = ast.get_source_segment(candidate_source, candidate_filter)
    original_block = """            filtered_mean, filtered_cov, log_likelihood = _condition_on_diagonal(
                pred_mean, pred_cov, H, D, d, R, u, y)
            ll += log_likelihood"""
    ablation_block = """            filtered_mean, filtered_cov, _unused_owned_likelihood = _condition_on_diagonal(
                pred_mean, pred_cov, H, D, d, R, u, y)
            ll += _audit_baseline_log_likelihood(pred_mean, pred_cov, H, D, d, R, u, y)"""
    assert original_filter.count(original_block) == 1, "Candidate filter changed; audit the replacement before benchmarking"
    changed_filter = original_filter.replace(original_block, ablation_block, 1)
    assert candidate_source.count(original_filter) == 1
    source = candidate_source.replace(original_filter, changed_filter, 1)
    source += "\n\n# Original frozen-baseline likelihood; only the function name is changed.\n"
    source += "from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiagPlusLowRankCovariance as MVNLowRank\n\n"
    source += likelihood + "\n"
    ast.parse(source)
    return source, likelihood


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--source", type=Path, help="Candidate inference.py; defaults to the explicit worktree")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_ROOT / "baseline_inference.py")
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "likelihood_ablation")
    parser.add_argument("--warm-repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=5098927)
    args = parser.parse_args()
    assert args.warm_repeats >= 21
    worktree = args.worktree.resolve()
    source_path = args.source.resolve() if args.source is not None else worktree / "dynamax/linear_gaussian_ssm/inference.py"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(worktree))
    harness = load_module("audit_ablation_harness", args.harness.resolve())
    jax, np = harness.jax, harness.np
    current_source = source_path.read_text()
    baseline_source = args.baseline.read_text()
    modified_source, original_likelihood = make_ablation_source(current_source, baseline_source)
    candidate_snapshot = output_dir / "candidate_inference_snapshot.py"
    ablation_snapshot = output_dir / "optimized_conditioning_tfp_likelihood.py"
    candidate_snapshot.write_text(current_source)
    ablation_snapshot.write_text(modified_source)
    candidate = load_module("audit_shared_owned_likelihood", candidate_snapshot)
    ablation = load_module("audit_optimized_conditioning_tfp_likelihood", ablation_snapshot)
    modules = {"shared_owned_kernel": candidate, "optimized_conditioning_tfp_likelihood": ablation}
    report = {"metadata": {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": harness.platform.platform(),
        "versions": {name: harness.metadata.version(name) for name in ["jax", "jaxlib", "tfp-nightly", "numpy"]},
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "dtype": "float32", "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "compilation_cache": bool(jax.config.jax_enable_compilation_cache),
        "thread_environment": {name: harness.os.environ.get(name) for name in ["XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "JAX_PLATFORMS"]},
        "worktree": str(worktree), "candidate_source": str(source_path),
        "candidate_source_sha256": hashlib.sha256(current_source.encode()).hexdigest(),
        "baseline_source": str(args.baseline.resolve()),
        "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "ablation_source_sha256": hashlib.sha256(modified_source.encode()).hexdigest(),
        "extracted_baseline_likelihood": original_likelihood,
        "candidate_snapshot": str(candidate_snapshot), "ablation_snapshot": str(ablation_snapshot),
        "seed": args.seed, "warm_repeats": args.warm_repeats,
        "scope": "Matched full filter outputs, both outer-jitted with dynamic params/observations/inputs; optimized conditioning identical in both variants",
        "compiler_caveat": "The ablation ignores the owned helper likelihood. XLA dead-code elimination and common-subexpression elimination are intentionally permitted. This measures the remaining compiled effect of separate original TFP likelihood versus shared owned likelihood, not dependency deletion or import costs.",
        "attribution": "Compare to previously measured frozen baseline separately; gains here isolate likelihood/shared-work changes after eliminating observation-space square conditioning.",
        "warm_timing": "21 or more rounds; alternating variant order; synchronized full outputs; correctness checked first; three untimed warmups",
    }, "results": []}
    rng = random.Random(args.seed)
    for length, state_dim, emission_dim in [(32, 8, 1024), (32, 16, 2048)]:
        case_id = f"T{length}_K{state_dim}_D{emission_dim}"
        data = harness.make_data(length, state_dim, emission_dim, args.seed)
        observations, inputs = jax.device_put(data["observations"]), jax.device_put(data["inputs"])
        order = list(modules); rng.shuffle(order)
        executables, call_args, initial = {}, {}, {}
        record = {"id": case_id, "T": length, "K": state_dim, "D": emission_dim,
                  "data_sha256": harness.data_fingerprint(data), "compile_order": order, "variants": {}}
        for name in order:
            module = modules[name]
            call_args[name] = harness.ready((harness.params_for(module, data), observations, inputs))
            jax.clear_caches()
            before = time.perf_counter_ns()
            lowered = jax.jit(module.lgssm_filter).lower(*call_args[name])
            lower_ms = (time.perf_counter_ns() - before) / 1e6
            before = time.perf_counter_ns()
            executable = lowered.compile()
            compile_ms = (time.perf_counter_ns() - before) / 1e6
            executables[name] = executable
            hlo = harness.inspect_hlo(executable, case_id, name, length, state_dim, emission_dim, output_dir)
            assert hlo["observation_space_square_shape_match_count"] == 0, (name, hlo["shape_matches"])
            record["variants"][name] = {"lower_ms": lower_ms, "compile_ms": compile_ms,
                "lower_plus_compile_ms": lower_ms + compile_ms,
                "compiled_memory": harness.memory_record(executable), "optimized_hlo": hlo}
            initial[name] = harness.posterior_record(harness.ready(executable(*call_args[name])))
        oracle = harness.numpy_information_reference(data)
        record["oracle_accuracy"] = {name: harness.compare_posteriors(
            oracle, posterior, 1e-6, 1e-4, enforce=True, loglik_atol=1e-3, loglik_rtol=5e-5)
            for name, posterior in initial.items()}
        record["between_variant_correctness"] = harness.compare_posteriors(
            initial["shared_owned_kernel"], initial["optimized_conditioning_tfp_likelihood"],
            1e-6, 1e-4, enforce=True, loglik_atol=1e-3, loglik_rtol=5e-5)
        for _ in range(3):
            for name in modules:
                harness.ready(executables[name](*call_args[name]))
        samples = {name: [] for name in modules}
        start_order = list(modules); rng.shuffle(start_order)
        record["warm_round_orders"] = []
        for repeat in range(args.warm_repeats):
            round_order = start_order if repeat % 2 == 0 else start_order[::-1]
            record["warm_round_orders"].append(round_order)
            for name in round_order:
                before = time.perf_counter_ns()
                harness.ready(executables[name](*call_args[name]))
                samples[name].append((time.perf_counter_ns() - before) / 1e6)
        for name, values in samples.items():
            record["variants"][name].update(warm_ms=values, warm_median_ms=statistics.median(values),
                warm_p10_ms=float(np.percentile(values, 10)), warm_p90_ms=float(np.percentile(values, 90)))
        record["shared_owned_over_separate_tfp_speedup"] = (
            record["variants"]["optimized_conditioning_tfp_likelihood"]["warm_median_ms"] /
            record["variants"]["shared_owned_kernel"]["warm_median_ms"])
        report["results"].append(record)
        (output_dir / "likelihood_ablation_results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(case_id, {name: round(record["variants"][name]["warm_median_ms"], 4) for name in modules},
              "owned speedup", round(record["shared_owned_over_separate_tfp_speedup"], 3), flush=True)
    print(f"Wrote {output_dir / 'likelihood_ablation_results.json'}", flush=True)


if __name__ == "__main__":
    main()
