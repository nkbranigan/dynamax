"""Sample the two notebook models for a given number of timesteps, with lindermanlab/ssm.

Usage: <python with lindermanlab/ssm installed> make_data.py <slds|rslds> <num_timesteps>
Writes scratch_data/<name>_T<num_timesteps>.npz, keyed as the recovery test's fixture: recovery/<name>/y (emissions,
(num_sequences, sequence length, N) float64), z and x (the true states and latents, the sequences joined end to end)
and true/<group> (the true parameters in the test's natural form).
The models are those of ssm's notebooks 3-Switching-Linear-Dynamical-System.ipynb and 4-Recurrent-SLDS.ipynb (which
docs/notebooks/slds/{slds,rslds}.ipynb port), built by the recovery test's fixture script with the notebooks' own
code and seeds and sampled by ssm, so the benchmark data do not depend on dynamax. The first sequence follows the
notebook's RNG state, so slds_T1000 (drawn after the notebook's 100-step sample) and rslds_T10000 are the ssm
notebooks' own datasets.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def path(name, num_timesteps):
    return os.path.join(HERE, "..", "..", "scratch_data", f"{name}_T{num_timesteps}.npz")


# Datasets longer than the notebook's are batches of independent sequences of the notebook's length. The rSLDS needs
# this: a single long NASCAR sequence drifts away from the track (max |x| about 7 at T=10k, the notebook's length,
# far more at 100k and 1M). The SLDS follows the same rule.
SEQUENCE_LENGTH = {"slds": 1000, "rslds": 10000}


def sample(model, name, num_timesteps):
    """(z, x, y) for independent sequences of SEQUENCE_LENGTH steps: z and x joined end to end, y stacked."""
    length = SEQUENCE_LENGTH[name]
    assert num_timesteps % length == 0, f"num_timesteps must be a multiple of {length} for {name}"
    zs, xs, ys = zip(*(model.sample(length) for _ in range(num_timesteps // length)))
    for i, x in enumerate(xs):
        assert name != "rslds" or np.abs(x).max() < 10, f"sequence {i} left the track: max|x| = {np.abs(x).max():.2f}"
    return np.concatenate(zs), np.concatenate(xs), np.stack(ys)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "..", "dynamax", "slds", "fixtures"))
    from export_ssm_fixtures import NOTEBOOKS, natural_params
    name, num_timesteps = sys.argv[1], int(sys.argv[2])
    model = NOTEBOOKS[name][0]()[0]
    z, x, y = sample(model, name, num_timesteps)
    out = {"y": np.asarray(y, dtype=np.float64), "z": np.asarray(z), "x": np.asarray(x, dtype=np.float64),
           **{f"true/{k}": np.asarray(v) for k, v in natural_params(model).items()}}
    os.makedirs(os.path.dirname(path(name, num_timesteps)), exist_ok=True)
    np.savez(path(name, num_timesteps), **{f"recovery/{name}/{k}": v for k, v in out.items()})
    print(path(name, num_timesteps), y.shape, "finite:", bool(np.isfinite(y).all()),
          "max|x|:", float(np.abs(x).max()), "states:", np.bincount(np.asarray(z)).tolist())
