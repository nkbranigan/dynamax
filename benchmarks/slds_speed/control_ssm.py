"""Control for the recovery plot: lindermanlab/ssm with random parameters and no learning.

Usage: <python with lindermanlab/ssm installed> control_ssm.py <slds|rslds> <num_timesteps>
Prints one JSON line. The parameters are those ssm's constructor draws at random, except the transition matrix: ssm
draws it as 0.95 I plus noise, which is also how the true one was drawn, so each of its rows is drawn from a flat
Dirichlet instead, as dynamax's prior does. The seed differs from the seeds that built the true models (0 and 12345),
which would reproduce some of the true parameters. The states are inferred with approximate_posterior at its
defaults and scored like a fit.
"""
import json
import os
import sys
import warnings

import numpy as np
import numpy.random as npr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "dynamax", "slds", "fixtures"))
from export_ssm_fixtures import NOTEBOOKS, natural_params  # noqa: E402
from recovery import scores  # noqa: E402

SEED = 1

if __name__ == "__main__":
    name, num_timesteps = sys.argv[1], int(sys.argv[2])
    saved = np.load(os.path.join(HERE, "..", "..", "scratch_data", f"{name}_T{num_timesteps}.npz"))
    data = saved["emissions"]
    _, fit_model = NOTEBOOKS[name][0]()
    npr.seed(SEED)
    model = fit_model()
    if hasattr(model.transitions, "log_Ps"):
        model.transitions.log_Ps = np.log(npr.dirichlet(np.ones(model.K), size=model.K))
    _, q = model.approximate_posterior([data], method="laplace_em", variational_posterior="structured_meanfield", verbose=0)
    x_hat = q.mean_continuous_states[0]
    z_hat = model.most_likely_states(x_hat, data)
    print(json.dumps({
        "library": "ssm", "dtype": "float64", "dataset": name, "num_timesteps": num_timesteps, "control": True,
        **scores(name, saved, natural_params(model), x_hat, z_hat),
    }))
