"""Score a fit with the recovery test's own function (dynamax/slds/ssm_recovery_test.py), so that the benchmark and
the test report the same quantities."""
from dynamax.slds.ssm_recovery_test import recovery


def scores(name, saved, params, x_hat, z_hat):
    """The test's scores of one fit on a dataset written by make_data.py: discrete state recovery, continuous state
    recovery and parameter recovery per group. ``params`` are the fitted parameters in the test's natural form."""
    truth = {f"recovery/{name}/x": saved["latents"], f"recovery/{name}/z": saved["states"],
             **{f"recovery/{name}/{k}": saved[k] for k in saved.files if k.startswith("true/")}}
    out = recovery(name, params, x_hat, z_hat, truth)
    return {"state_accuracy": out.pop("discrete state recovery"), "latent_r2": out.pop("continuous state recovery"),
            "parameter_recovery": out}
