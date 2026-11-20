We benchmark SLDS and rSLDS fitting in Dynamax against [`lindermanlab/ssm`](https://github.com/lindermanlab/ssm): speed, and recovery of the true states and parameters.

To reproduce the results and the plots, from the repository root, with each machine's name passed as `--name`:

```bash
# environment with Dynamax and ssm (the same one as dynamax/slds/fixtures/README.md)
conda create -n dyna_dev_ssm python=3.10 -y && conda activate dyna_dev_ssm
pip install -e '.[dev]'
pip install cython
pip install --no-build-isolation "git+https://github.com/lindermanlab/ssm.git@eb6c8aa"
pip install "jax[cuda12]"                 # optional: GPU support, for bench_dynamax.py --device gpu

python benchmarks/slds_vs_ssm/make_data.py                               # samples the nine datasets with ssm
python benchmarks/slds_vs_ssm/bench_ssm.py --name epyc_7502              # writes cpu_epyc_7502_ssm_*.jsonl
python benchmarks/slds_vs_ssm/bench_dynamax.py --name epyc_7502          # writes cpu_epyc_7502_dynamax_*.jsonl
python benchmarks/slds_vs_ssm/bench_dynamax.py --device gpu --name h200  # on a machine with a GPU; writes gpu_h200_dynamax_*.jsonl
python benchmarks/slds_vs_ssm/plot.py                                    # writes speed_*.png and speed_total_*.png for each CPU, and recovery_*.png for one
```
