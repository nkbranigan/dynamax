"""Sample the two notebook models for a given number of timesteps, with lindermanlab/ssm.

Usage: <python with ssm installed> make_data.py <slds|rslds> <num_timesteps> [sampling seed]
Writes scratch_data/<name>_T<num_timesteps>.npz with emissions (T, N) float64, the true states and latents, and the
true parameters in the recovery test's natural form (true/<group>).
The models are those of ssm's notebooks 3-Switching-Linear-Dynamical-System.ipynb and 4-Recurrent-SLDS.ipynb (which
docs/notebooks/slds/{slds,rslds}.ipynb port), built by the recovery test's fixture script with the notebooks' own
code and seeds and sampled by ssm, so the benchmark data do not depend on dynamax. Without a sampling seed the first sample follows the notebook's RNG
state, so slds_T1000 (drawn after the notebook's 100-step sample) and rslds_T10000 are the ssm notebooks' own
datasets; other lengths are longer samples of the same models. Only path() is imported by the bench scripts, so ssm
is imported lazily.
"""
import os
import sys

import numpy as np


def path(name, num_timesteps):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scratch_data", f"{name}_T{num_timesteps}.npz")


def fixtures():
    """The recovery test's export script (dynamax/slds/fixtures/export_ssm_fixtures.py), whose builders of the two
    notebook models and natural_params are reused here. Imported lazily: it imports ssm."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dynamax", "slds", "fixtures"))
    import export_ssm_fixtures
    return export_ssm_fixtures


# A single long NASCAR sample drifts away from the track (max |x| about 7 at T=10k, the notebook's length, far more
# at 100k and 1M), so longer datasets are stitched from independent 10k-step samples.
SEGMENT = {"rslds": 10000}


def sample(name, num_timesteps, seed=None):
    """One sequence (z, x, y); for models in SEGMENT, a concatenation of independent SEGMENT-step samples."""
    import numpy.random as npr
    model = fixtures().NOTEBOOKS[name][0]()[0]
    if seed is not None:
        npr.seed(seed)
    seg = SEGMENT.get(name, num_timesteps)
    if num_timesteps <= seg:
        return model.sample(num_timesteps)
    assert num_timesteps % seg == 0, f"num_timesteps must be a multiple of {seg} for {name}"
    pieces = [model.sample(seg) for _ in range(num_timesteps // seg)]
    for i, (_, x, _) in enumerate(pieces):
        assert np.abs(x).max() < 10, f"segment {i} left the track: max|x| = {np.abs(x).max():.2f}"
    return tuple(np.concatenate([p[j] for p in pieces]) for j in range(3))


if __name__ == "__main__":
    name, num_timesteps = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    states, latents, emissions = sample(name, num_timesteps, seed)
    true_model = fixtures().NOTEBOOKS[name][0]()[0]
    true = {f"true/{k}": np.asarray(v) for k, v in fixtures().natural_params(true_model).items()}
    os.makedirs(os.path.dirname(path(name, num_timesteps)), exist_ok=True)
    np.savez(path(name, num_timesteps), emissions=np.asarray(emissions, dtype=np.float64),
             states=np.asarray(states), latents=np.asarray(latents, dtype=np.float64), **true)
    print(path(name, num_timesteps), emissions.shape, "finite:", bool(np.isfinite(emissions).all()),
          "max|x|:", float(np.abs(latents).max()), "states:", np.bincount(np.asarray(states)).tolist())
