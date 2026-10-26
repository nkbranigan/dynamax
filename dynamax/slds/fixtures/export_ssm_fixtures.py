"""
Run ``lindermanlab/ssm`` (commit eb6c8aa) and save its outputs to ``ssm_fixtures.npz`` for
``dynamax/slds/ssm_fit_steps_test.py`` (the ``steps/`` entries: EM sub-steps) and
``dynamax/slds/ssm_recovery_test.py`` (the ``recovery/`` entries: fits of the notebook datasets). Runs in an
environment with ``ssm`` installed (see README.md); dynamax is not needed.

    python export_ssm_fixtures.py
"""
import copy
import os

import numpy as np
import numpy.random as npr

import ssm
import ssm.variational as variational
from ssm.util import random_rotation

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssm_fixtures.npz")


# ---- EM sub-steps: the steps/ entries of ssm_fixtures.npz ----------------------------------------------------------
K, D, N, T = 3, 2, 8, 100   # N > D * K: ssm's multi-subspace PCA needs D * K components


def transition_arrays(transitions):
    """ssm transition parameters as plain arrays, by class."""
    name = type(transitions).__name__
    if name == "StationaryTransitions":
        return dict(transition_matrix=transitions.transition_matrix)
    if name == "RecurrentTransitions":
        return dict(log_Ps=transitions.log_Ps, Rs=transitions.Rs, Ws=transitions.Ws)
    if name == "RecurrentOnlyTransitions":
        return dict(r=transitions.r, Rs=transitions.Rs, Ws=transitions.Ws)
    raise ValueError(name)


def export_case(name, transitions, single_subspace, seed, out):
    """One case: parameters, data, one update of q(z), one update of q(x), and the M-steps."""
    npr.seed(seed)
    slds = ssm.SLDS(N, K, D, transitions=transitions, dynamics="gaussian", emissions="gaussian", single_subspace=single_subspace)
    _, _, y = slds.sample(T)
    datas, inputs, masks, tags = [y], [np.zeros((T, 0))], [np.ones_like(y, dtype=bool)], [None]
    slds.initialize(datas, inputs, masks, tags, num_init_iters=10)

    put = lambda key, value: out.__setitem__(f"{name}/{key}", np.asarray(value))
    put("y", y)
    put("pi0", slds.init_state_distn.initial_state_distn)
    for key, value in transition_arrays(slds.transitions).items():
        put(f"trans/{key}", value)
    dyn = slds.dynamics
    put("dyn/As", dyn.As); put("dyn/bs", dyn.bs); put("dyn/Vs", dyn.Vs); put("dyn/Sigmas", dyn.Sigmas)
    put("dyn/mu_init", dyn.mu_init); put("dyn/Sigmas_init", dyn.Sigmas_init)
    emis = slds.emissions
    put("emis/Cs", emis.Cs); put("emis/ds", emis.ds); put("emis/Fs", emis.Fs); put("emis/inv_etas", emis.inv_etas)

    # The starting q(x) is N(invert(y), 1e-2 I). One update of q(z) (from one x sample), then one update of q(x)
    # by Newton's method. The x sample is the update's first random draw, so it is read by drawing it here and
    # then rewinding numpy's random state.
    q = variational.SLDSStructuredMeanFieldVariationalPosterior(slds, datas, inputs, masks, tags)
    state = npr.get_state()
    x_sample = q.sample_continuous_states()[0]
    npr.set_state(state)
    slds._fit_laplace_em_discrete_state_update(q, datas, inputs, masks, tags, num_samples=1)
    put("x_sample", x_sample)
    params0 = q.continuous_state_params[0]
    for key in ["J_ini", "J_dyn_11", "J_dyn_21", "J_dyn_22", "J_obs", "h_ini", "h_dyn_1", "h_dyn_2", "h_obs"]:
        put(f"q0/{key}", params0[key])
    dparams = q.discrete_state_params[0]
    Ez, Ezzp1, log_Z = q.discrete_expectations[0]
    # ssm keeps stationary transitions as one (1, K, K) matrix with time-summed pairwise expectations; dynamax takes (K, K)
    squeeze = (lambda a: a[0]) if transitions == "standard" else (lambda a: a)
    put("qz/pi0", dparams["pi0"]); put("qz/Ps", squeeze(dparams["Ps"])); put("qz/log_likes", dparams["log_likes"])
    put("qz/Ez", Ez); put("qz/Ezzp1", squeeze(Ezzp1)); put("qz/log_Z", log_Z)

    # Hessian blocks, h = J x, and the objective at the x sample.
    J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs = slds._laplace_neg_hessian_params(y, inputs[0], masks[0], None, x_sample, Ez, Ezzp1)
    h_ini, h_dyn_1, h_dyn_2, h_obs = slds._laplace_neg_hessian_params_to_hs(x_sample, J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs)
    for key, value in dict(J_ini=J_ini, J_dyn_11=J_dyn_11, J_dyn_21=J_dyn_21, J_dyn_22=J_dyn_22, J_obs=J_obs,
                           h_ini=h_ini, h_dyn_1=h_dyn_1, h_dyn_2=h_dyn_2, h_obs=h_obs).items():
        put(f"hess/{key}", value)
    put("hess/neg_expected_log_joint", slds._laplace_neg_expected_log_joint(y, inputs[0], masks[0], None, x_sample, Ez, Ezzp1, scale=1))

    slds._fit_laplace_em_continuous_state_update(q, datas, inputs, masks, tags, continuous_optimizer="newton",
                                                 continuous_tolerance=1e-10, continuous_maxiter=200)
    params1 = q.continuous_state_params[0]
    for key in ["J_ini", "J_dyn_11", "J_dyn_21", "J_dyn_22", "J_obs", "h_ini", "h_dyn_1", "h_dyn_2", "h_obs"]:
        put(f"q1/{key}", params1[key])
    log_Z, Ex, Sigmas, ExxnT = q.continuous_expectations[0]      # kalman_info_smoother outputs of the updated q(x)
    put("q1/log_Z", log_Z); put("q1/Ex", Ex); put("q1/Sigmas", Sigmas); put("q1/ExxnT", ExxnT)
    put("q1/continuous_entropy", q._continuous_entropy())     # ssm's entropy of q(z) is biased by the 1e-16 inside its logs

    # M-steps on copies: the initial state with the discrete expectations, the transitions with the x sample, the
    # dynamics with the exact continuous expectations, and the emissions with the x sample.
    xmasks = [np.ones_like(x_sample, dtype=bool)]
    init = copy.deepcopy(slds.init_state_distn)
    init.m_step(q.discrete_expectations, [x_sample], inputs, xmasks, tags)
    put("mstep/pi0", init.initial_state_distn)
    trans = copy.deepcopy(slds.transitions)
    if transitions == "standard":
        trans.m_step(q.discrete_expectations, [x_sample], inputs, xmasks, tags)
    else:
        # run to convergence (ssm's default tol is 1e-4) so that the test can compare the two libraries' optima
        trans.m_step(q.discrete_expectations, [x_sample], inputs, xmasks, tags, num_iters=1000, tol=1e-12)
        # the expected log transition term alone (ssm's Dirichlet log prior omits the normalizing constant)
        put("mstep/trans/expected_log_trans", np.sum(Ezzp1 * trans.log_transition_matrices(x_sample, inputs[0], xmasks[0], None)))
    for key, value in transition_arrays(trans).items():
        put(f"mstep/trans/{key}", value)
    dyn = copy.deepcopy(slds.dynamics)
    dyn.m_step(q.discrete_expectations, [x_sample], inputs, xmasks, tags, continuous_expectations=q.continuous_expectations)
    put("mstep/As", dyn.As); put("mstep/bs", dyn.bs); put("mstep/Sigmas", dyn.Sigmas)
    emis = copy.deepcopy(slds.emissions)
    emis.m_step(q.discrete_expectations, [x_sample], datas, inputs, masks, tags)
    put("mstep/Cs", emis.Cs); put("mstep/ds", emis.ds); put("mstep/inv_etas", emis.inv_etas)
    print(f"{name}: exported ({len(out)} arrays so far)")


def export_fit_steps(out):
    """The EM sub-steps of the six cases, under ``steps/<transitions>/<subspaces>/``."""
    seed = 0
    for transitions in ["standard", "recurrent", "recurrent_only"]:
        for single_subspace in [True, False]:
            export_case(f"steps/{transitions}/{'single' if single_subspace else 'multi'}", transitions, single_subspace, seed, out)
            seed += 1


# ---- recovery: the recovery/ entries of ssm_fixtures.npz -----------------------------------------------------------
def notebook_slds():
    """The SLDS of ssm's notebook 3 (ported by docs/notebooks/slds/slds.ipynb), built with the notebook's own code
    and seed, and the model ssm fits to it at its defaults. The notebook's 100-step visualization sample is drawn
    here so that the next sample is its 1000-step fitting data."""
    npr.seed(0)
    num_states, state_dim, emission_dim = 5, 2, 10
    true = ssm.SLDS(emission_dim, num_states, state_dim, emissions="gaussian_orthog")
    for k in range(num_states):
        true.dynamics.As[k] = 0.95 * random_rotation(state_dim, theta=(k + 1) * np.pi / 20)
    true.sample(100)
    fit_model = lambda: ssm.SLDS(emission_dim, num_states, state_dim, transitions="standard", dynamics="gaussian",
                                 emissions="gaussian", single_subspace=True)
    return true, fit_model


def notebook_rslds():
    """The NASCAR rSLDS of ssm's notebook 4 (ported by docs/notebooks/slds/rslds.ipynb), built with the notebook's
    own code and seed, and the model ssm fits to it at its defaults."""
    npr.seed(12345)
    num_states, state_dim, emission_dim = 4, 2, 10
    As = [random_rotation(state_dim, np.pi / 24.), random_rotation(state_dim, np.pi / 48.)]
    centers = [np.array([+2.0, 0.]), np.array([-2.0, 0.])]
    bs = [-(A - np.eye(state_dim)).dot(center) for A, center in zip(As, centers)]
    As.append(np.eye(state_dim)); bs.append(np.array([+0.1, 0.]))
    As.append(np.eye(state_dim)); bs.append(np.array([-0.25, 0.]))
    w1, b1 = np.array([+1.0, 0.0]), np.array([-2.0])
    w2, b2 = np.array([-1.0, 0.0]), np.array([-2.0])
    w3, b3 = np.array([0.0, +1.0]), np.array([0.0])
    w4, b4 = np.array([0.0, -1.0]), np.array([0.0])
    Rs = np.vstack((100 * w1, 100 * w2, 10 * w3, 10 * w4))
    r = np.concatenate((100 * b1, 100 * b2, 10 * b3, 10 * b4))
    true = ssm.SLDS(emission_dim, num_states, state_dim, transitions="recurrent_only", dynamics="diagonal_gaussian",
                    emissions="gaussian_orthog", single_subspace=True)
    true.dynamics.mu_init = np.tile(np.array([[0, 1]]), (num_states, 1))
    true.dynamics.sigmasq_init = 1e-4 * np.ones((num_states, state_dim))
    true.dynamics.As = np.array(As)
    true.dynamics.bs = np.array(bs)
    true.dynamics.sigmasq = 1e-4 * np.ones((num_states, state_dim))
    true.transitions.Rs = Rs
    true.transitions.r = r
    true.emissions.inv_etas = np.log(1e-2) * np.ones((1, emission_dim))
    fit_model = lambda: ssm.SLDS(emission_dim, num_states, state_dim, transitions="recurrent_only",
                                 dynamics="diagonal_gaussian", emissions="gaussian", single_subspace=True)
    return true, fit_model


NOTEBOOKS = {"slds": (notebook_slds, 1000), "rslds": (notebook_rslds, 10000)}   # the notebooks' sequence lengths


def natural_params(model):
    """An ssm SLDS's parameters as probabilities and variances, named by the letters of dynamax's model docstrings."""
    out = {"F": model.dynamics.As, "b": model.dynamics.bs, "Q": model.dynamics.Sigmas, "H": model.emissions.Cs[0],
           "d": model.emissions.ds[0], "R": np.exp(model.emissions.inv_etas[0])}
    if hasattr(model.transitions, "log_Ps"):
        out["P"] = model.transitions.transition_matrix
    else:
        out["L"], out["W"] = model.transitions.r, model.transitions.Rs
    return out


def export_recovery(out, num_iters=100):
    """ssm's fits of the two notebook datasets, under ``recovery/<model>/``: the ssm notebook's own dataset, the
    true parameters, and, after a fit at ssm's defaults (``num_iters`` of Laplace-EM), ssm's parameters, posterior
    mean of x, and most likely states."""
    for name, (build, num_timesteps) in NOTEBOOKS.items():
        true, fit_model = build()
        z, x, y = true.sample(num_timesteps)
        npr.seed(0)
        model = fit_model()
        model.initialize([y])
        elbos, q = model.fit([y], method="laplace_em", variational_posterior="structured_meanfield",
                             num_iters=num_iters, initialize=False, verbose=0)
        x_hat = q.mean_continuous_states[0]
        z_hat = model.most_likely_states(x_hat, y)
        out[f"recovery/{name}/y"] = y; out[f"recovery/{name}/z"] = z; out[f"recovery/{name}/x"] = x
        out[f"recovery/{name}/x_hat"] = x_hat; out[f"recovery/{name}/z_hat"] = z_hat
        for k, v in natural_params(true).items():
            out[f"recovery/{name}/true/{k}"] = v
        for k, v in natural_params(model).items():
            out[f"recovery/{name}/ssm/{k}"] = v
        print(f"{name}: exported, final elbo {elbos[-1]:.1f}", flush=True)


if __name__ == "__main__":
    out = {}
    export_fit_steps(out)
    export_recovery(out)
    np.savez(FIXTURE, **out)
    print(f"wrote {FIXTURE}")
