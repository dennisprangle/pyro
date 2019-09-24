import argparse
import datetime
from contextlib import ExitStack
import math
import subprocess
import pickle
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
    return 0.5 * (term1 + term2).item()


def optimal_eig(alpha, D, sigmasq, S=100 * 1000):
    delta = torch.arange(S + 1).float() / float(S)
    epsilon = 1.0 - delta
    max_eig, optimal_delta = analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq).max(dim=-1)

    for zoom in [0.01, 0.001, 0.0001, 0.00001]:
        delta = delta[optimal_delta].item() - zoom + 2.0 * zoom * torch.arange(S + 1).float() / float(S)
        epsilon = 1.0 - delta
        max_eig, optimal_delta = analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq).max(dim=-1)

    return max_eig.item(), delta[optimal_delta].item(), epsilon[optimal_delta].item()


def get_prior_precision(alpha, D):
    u = torch.ones(D).unsqueeze(-1)
    u[0:int(D/2), 0] = alpha
    prior_precision = (1.0 + 1.0 / D) * torch.eye(D) - (1.0 / D) * torch.mm(u, u.t())
    return prior_precision


def get_prior_scale_tril(alpha, D):
    precision = get_prior_precision(alpha, D)
    return precision.cholesky()


def total_budget(D):
    return 0.5 * D


def make_model(D=4, alpha=0.5, sigma=1.0):
    def init_budget_fn():
        b = torch.rand(D)
        return b / b.sum()

    def model_learn_design(design_prototype):
        B = total_budget(D) * pyro.param('budget', init_budget_fn, constraint=constraints.simplex)
        batch_shape = design_prototype.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(D), precision_matrix=get_prior_precision(alpha, D)))
            return pyro.sample("y", dist.Normal(x * B, sigma * B.sqrt()).to_event(1))
    return model_learn_design


def make_posterior_guide(d, alpha, D):
    def posterior_guide(y_dict, design, observation_labels, target_labels, **kwargs):
        y = torch.cat(list(y_dict.values()), dim=-1)
        A = pyro.param("A", lambda: torch.zeros(d, 1, D))
        scale_tril = pyro.param("scale_tril", lambda: lexpand(get_prior_scale_tril(alpha, D), d),
                                constraint=torch.distributions.constraints.lower_cholesky)
        pyro.sample("x", dist.MultivariateNormal(A * y, scale_tril=scale_tril))
    return posterior_guide


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim):
    params = None
    est_loss_history = []
    xi_history = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples)
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


def main(num_steps, experiment_name, estimators, seed, start_lr, end_lr):
    output_dir = "./"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'
    estimators = estimators.split(",")

    alpha = 0.5
    D = 12
    sigma = 1.0
    sigmasq = sigma ** 2

    model_learn_design = make_model(D=D, alpha=alpha, sigma=sigma)

    for estimator in estimators:
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)

        if estimator == 'posterior':
            guide = make_posterior_guide(1, alpha, D)
            loss = _differentiable_posterior_loss(model_learn_design, guide, "y", "x")

        elif estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: nce_eig(model=model_learn_design, design=d, observation_labels="y",
                                                      target_labels="x", N=N, M=10, **kwargs)
            loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            guide = make_posterior_guide(1, alpha, D)
            eig_loss = _ace_eig_loss(model_learn_design, guide, 10, "y", "x")
            loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr/start_lr)**(1/num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, 1, D)  # this is annoying, code needs refactor

        xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=10,
                                                             num_steps=num_steps, optim=scheduler)

        opt_eig, opt_delta, opt_epsilon = optimal_eig(alpha, D, sigmasq)
        opt_design = opt_delta * torch.ones(D)
        opt_design[int(D/2):] = opt_epsilon

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history}

        initial_design = total_budget(D) * xi_history[0]
        final_design = total_budget(D) * pyro.param('budget')
        initial_eig = analytic_eig_B(alpha, D, initial_design, sigmasq)
        final_eig = analytic_eig_B(alpha, D, final_design, sigmasq)

        print("Initial/Final/Optimal EIG:  %.5f / %.5f / %.5f" % (initial_eig, final_eig, opt_eig))
        print("Initial design:\n", initial_design.data.numpy())
        print("Final design:\n", final_design.data.numpy())
        print("Optimal design:\n", opt_design.data.numpy())
        print("Mean Absolute EIG Error: %.5f" % math.fabs(final_eig - opt_eig))
        print("Mean Absolute Design Error: %.5f" % (final_design - opt_design).abs().sum())

        #with open(results_file, 'wb') as f:
        #    pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=5000, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.1, type=float)
    parser.add_argument("--end-lr", default=0.0001, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
