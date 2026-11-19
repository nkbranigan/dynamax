We benchmark SLDS and rSLDS fitting in Dynamax against [`lindermanlab/ssm`](https://github.com/lindermanlab/ssm): speed, and recovery of the true states and parameters. Each library and device has its own results file: `cpu_ssm.jsonl`, `cpu_dynamax.jsonl`, and, when Dynamax is also run on a GPU, `gpu_dynamax.jsonl`.

To reproduce the results and the three plots, from the repository root:

```bash
# environment with Dynamax and ssm (the same one as dynamax/slds/fixtures/README.md)
conda create -n dyna_dev_ssm python=3.10 -y && conda activate dyna_dev_ssm
pip install -e '.[dev]'
pip install cython
pip install --no-build-isolation "git+https://github.com/lindermanlab/ssm.git@eb6c8aa"
pip install "jax[cuda12]"                 # optional: GPU support, for bench_dynamax.py --device gpu

python benchmarks/slds_vs_ssm/make_data.py                    # samples the seven datasets with ssm
python benchmarks/slds_vs_ssm/bench_ssm.py                    # about 4.5 hours; writes cpu_ssm.jsonl
python benchmarks/slds_vs_ssm/bench_dynamax.py                # about 10 minutes; writes cpu_dynamax.jsonl
python benchmarks/slds_vs_ssm/bench_dynamax.py --device gpu   # optional, on a machine with a GPU; writes gpu_dynamax.jsonl
python benchmarks/slds_vs_ssm/plot.py                         # writes speed.png, speed_total.png and recovery.png
```
