We benchmark SLDS and rSLDS fitting in Dynamax against [`lindermanlab/ssm`](https://github.com/lindermanlab/ssm): speed, and recovery of the true states and parameters. `results.jsonl` holds the committed results.

To reproduce the results and the three plots, from the repository root:

```bash
# environment with Dynamax and ssm (the same one as dynamax/slds/fixtures/README.md)
conda create -n dyna_dev_ssm python=3.10 -y && conda activate dyna_dev_ssm
pip install -e '.[dev]'
pip install cython
pip install --no-build-isolation "git+https://github.com/lindermanlab/ssm.git@eb6c8aa"

# set PYTHON at the top of run.py to this environment's python (`which python`)
rm benchmarks/slds_vs_ssm/results.jsonl   # run.py skips fits already in the file
python benchmarks/slds_vs_ssm/run.py      # about 35 minutes; writes the datasets and results.jsonl
python benchmarks/slds_vs_ssm/plot.py     # writes speed.png, speed_total.png and recovery.png
```
