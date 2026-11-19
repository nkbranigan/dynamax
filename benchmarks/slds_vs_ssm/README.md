We benchmark SLDS and rSLDS fitting in Dynamax against [`lindermanlab/ssm`](https://github.com/lindermanlab/ssm): speed, and recovery of the true states and parameters. Each library and device writes one results file per model: `cpu_ssm_slds.jsonl` and `cpu_ssm_rslds.jsonl`, `cpu_dynamax_slds.jsonl` and `cpu_dynamax_rslds.jsonl`, and, when Dynamax is also run on a GPU, `gpu_dynamax_slds.jsonl` and `gpu_dynamax_rslds.jsonl`. `bench_ssm.py` and `bench_dynamax.py` take `--model slds` or `--model rslds` to fit one model alone, so the two models can run at the same time.

To reproduce the results and the three plots, from the repository root:

```bash
# environment with Dynamax and ssm (the same one as dynamax/slds/fixtures/README.md)
conda create -n dyna_dev_ssm python=3.10 -y && conda activate dyna_dev_ssm
pip install -e '.[dev]'
pip install cython
pip install --no-build-isolation "git+https://github.com/lindermanlab/ssm.git@eb6c8aa"
pip install "jax[cuda12]"                 # optional: GPU support, for bench_dynamax.py --device gpu

python benchmarks/slds_vs_ssm/make_data.py                    # samples the nine datasets with ssm
python benchmarks/slds_vs_ssm/bench_ssm.py                    # about 4.5 hours; writes cpu_ssm_*.jsonl
python benchmarks/slds_vs_ssm/bench_dynamax.py                # about 10 minutes; writes cpu_dynamax_*.jsonl
python benchmarks/slds_vs_ssm/bench_dynamax.py --device gpu   # optional, on a machine with a GPU; writes gpu_dynamax_*.jsonl
python benchmarks/slds_vs_ssm/plot.py                         # writes speed.png, speed_total.png and recovery.png
```
