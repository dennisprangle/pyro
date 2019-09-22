import argparse
import datetime
from contextlib import ExitStack
import math
import subprocess
import pickle
from functools import partial
import numpy as np

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.oed.eig import _eig_from_ape, nce_eig, _ace_eig_loss
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss
from pyro.contrib.oed.util import linear_model_ground_truth
from pyro.contrib.util import rmv, iter_plates_to_shape, lexpand, rvv, rexpand


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def prior_entropy(alpha, D):
    return analytic_eig_delta_epsilon(alpha, D, torch.tensor(0.0), torch.tensor(0.0), 1.0)


def analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq):
    oneplusd = 1.0 + 1.0 / D
    term1 = 0.5 * D * (oneplusd + delta / sigmasq).log()
    term2 = 0.5 * D * (oneplusd + epsilon / sigmasq).log()
    term3 = (1.0 - 0.5 * (alpha * alpha / (oneplusd + delta / sigmasq) + 1.0 / (oneplusd + epsilon / sigmasq))).log()
    return 0.5 * (term1 + term2 + term3)


def analytic_eig_B(alpha, D, B, sigmasq):
    oneplusd = 1.0 + 1.0 / D
    term1 = (oneplusd + B / sigmasq).log().sum()
    term2a = (alpha * alpha / (oneplusd + B[0:int(D/2)] / sigmasq)).sum()
    term2b = (1.0 / (oneplusd + B[int(D/2):] / sigmasq)).sum()
    term2 = (1.0 - (term2a + term2b) / D).log()
    return 0.5 * (term1 + term2)


def top_eig(alpha, D, sigmasq, S=1000):
    delta = torch.arange(S + 1).float() / float(S)
    epsilon = 1.0 - delta
    max_eig, optimal_delta = analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq).max(dim=-1)
    return max_eig.item(), delta[optimal_delta].item(), epsilon[optimal_delta].item()


def get_u(alpha, D):
    u = torch.ones(D)
    u[0:int(D/2)] = alpha
    return u.unsqueeze(-1) / math.sqrt(D)


def init_budget_fn(D):
    b = torch.rand(D)
    return b / b.sum()


def model_learn_design(design_prototype, D=4, alpha=0.5, sigma=1.0):
    total_budget = 0.5 * D
    B = total_budget * pyro.param('budget', partial(init_budget_fn, D=D), constraint=constraints.simplex)
    #xi = lexpand(], dim=-1), 1)
    batch_shape = design_prototype.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        u = get_u(alpha, D)
        prior_precision = (1.0 + 1.0 / D) * torch.eye(D) - torch.mm(u, u.t())

        x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(D), precision_matrix=prior_precision))
        return pyro.sample("y", dist.Normal(x * B, sigma * B.sqrt()).to_event(1))



def make_posterior_guide(d):
    def posterior_guide(y_dict, design, observation_labels, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        A = pyro.param("A", torch.zeros(d, 2, N))
        scale_tril = pyro.param("scale_tril", lexpand(prior_scale_tril, d),
                                constraint=torch.distributions.constraints.lower_cholesky)
        mu = rmv(A, y)
        pyro.sample("x", dist.MultivariateNormal(mu, scale_tril=scale_tril))
    return posterior_guide


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim, D, alpha, sigma):

    params = None
    est_loss_history = []
    xi_history = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, alpha=alpha, D=D, sigma=sigma)
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward()
        #agg_loss.backward(retain_graph=True)
        est_loss_history.append(loss.detach().clone())
        xi_history.append(pyro.param('budget').detach().clone())
        optim(params)
        optim.step()

    xi_history.append(pyro.param('budget').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def main(num_steps, experiment_name, estimators, seed, start_lr, end_lr, alpha=0.01):
    output_dir = "./"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'
    estimators = estimators.split(",")

    D=8
    sigma=0.25
    sigmasq=sigma ** 2

    for estimator in estimators:
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)

        # Fix correct loss
        #if estimator == 'posterior':
        #    guide = make_posterior_guide(1)
        #    loss = _differentiable_posterior_loss(model_learn_xi, guide, "y", "x")

        if estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: nce_eig(model=model_learn_design, design=d, observation_labels="y",
                                                      target_labels="x", N=N, M=10, **kwargs)
            loss = neg_loss(eig_loss)

        #elif estimator == 'ace':
        #    guide = make_posterior_guide(1)
        #    eig_loss = _ace_eig_loss(model_learn_xi, guide, 10, "y", "x")
        #    loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr/start_lr)**(1/num_steps)
        # optimizer = optim.Adam({"lr": start_lr})
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, 1, D)  # this is annoying, code needs refactor

        xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=10,
                                                             num_steps=num_steps, optim=scheduler,
                                                             alpha=alpha, D=D, sigma=sigma)

        #if estimator == 'posterior':
        #    est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, "x", est_loss_history, True, {})
        #else:
        est_eig_history = -est_loss_history

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history}

        print("est_eig_history mean",np.mean(np.array(est_eig_history[-100:])))
        print("est_eig_history mean",np.mean(np.array(est_eig_history[-200:])))
        print("est_eig_history mean",np.mean(np.array(est_eig_history[-500:])))
        print("top_eig", top_eig(alpha, D, sigmasq))
        print("first_eig", analytic_eig_B(alpha, D, 0.5 * D * xi_history[0], sigmasq))
        print("final_eig", analytic_eig_B(alpha, D, 0.5 * D * pyro.param('budget'), sigmasq))
        print("xi[0]", 0.5 * D * xi_history[0])
        print("xi[-1]", 0.5 * D * xi_history[-1])
        print("prior entropy", prior_entropy(alpha, D))

        #with open(results_file, 'wb') as f:
        #    pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=2000, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="nce", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.1, type=float)
    parser.add_argument("--end-lr", default=0.0001, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
