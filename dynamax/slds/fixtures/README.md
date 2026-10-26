# ssm oracle fixtures

`ssm_oracle_test.py` and `benchmark_test.py` in the parent directory compare dynamax's SLDS /
rSLDS Laplace variational EM with the reference implementation in
[`lindermanlab/ssm`](https://github.com/lindermanlab/ssm) at commit `eb6c8aa`. They read the
fixture files in this directory and **skip when the files are absent**, so CI never runs them and
no binary fixtures are committed (`*.npz` is ignored here).

To regenerate the fixtures locally:

```bash
# a Python 3.10 environment with ssm built from source (ssm needs numpy < 2 and cython)
python -m venv ssm_venv && ssm_venv/bin/pip install "numpy<2" scipy autograd scikit-learn tqdm cython
git clone https://github.com/lindermanlab/ssm && git -C ssm checkout eb6c8aa
ssm_venv/bin/pip install --no-build-isolation -e ssm
ssm_venv/bin/python dynamax/slds/fixtures/export_ssm_fixtures.py          # writes ssm_fixtures.npz
ssm_venv/bin/python dynamax/slds/fixtures/export_ssm_fixtures.py --benchmark   # writes ssm_benchmark.npz (slow)
```

`DYNAMAX_SSM_FIXTURES` and `DYNAMAX_SSM_BENCHMARK` override the paths the tests read.

`ssm_fixtures.npz` holds, for seeded ssm models (K = 3, D = 2, N = 8, T = 50; stationary,
recurrent and recurrent-only transitions; one shared or per-state emission subspaces; ssm's default
regularization and the dynamics regularization zeroed): the parameters after `initialize`, the
data, one set of discrete expectations, a fixed continuous trajectory, a q(x) sample and the noise
that produced it, and ssm's outputs of each deterministic sub-step (Hessian blocks, Newton mode,
smoother moments, q(z) update, dynamics and emission M-steps, recurrent M-step objective).
`ssm_benchmark.npz` holds ssm's final state accuracy and aligned latent RMSE over seeds on the
recovery problem of the tutorial notebook.
