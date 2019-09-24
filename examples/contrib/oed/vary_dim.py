import argparse
import datetime
from contextlib import ExitStack
import math
import subprocess
import pickle
import numpy as np
import pyro.contrib.gp as gp
import time

import torch
from torch.distributions import constraints
from torch.nn.functional import softmax

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


# find the exact EIG (up to a constant) for a design [delta, delta..., epsilon, epsilon, ...]
def analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq):
    oneplusd = 1.0 + 1.0 / D
    term1 = 0.5 * D * (oneplusd + delta / sigmasq).log()
    term2 = 0.5 * D * (oneplusd + epsilon / sigmasq).log()
    term3 = (1.0 - 0.5 * (alpha * alpha / (oneplusd + delta / sigmasq) + 1.0 / (oneplusd + epsilon / sigmasq))).log()
    return 0.5 * (term1 + term2 + term3)


# find the exact EIG (up to a constant) for a design B
def analytic_eig_budget(alpha, D, B, sigmasq):
    oneplusd = 1.0 + 1.0 / D
    term1 = (oneplusd + B / sigmasq).log().sum(-1)
    term2a = (alpha * alpha / (oneplusd + B[..., 0:int(D/2)] / sigmasq)).sum(-1)
    term2b = (1.0 / (oneplusd + B[..., int(D/2):] / sigmasq)).sum(-1)
    term2 = (1.0 - (term2a + term2b) / D).log()
    eig = 0.5 * (term1 + term2)
    if eig.numel() == 1:
        return eig.item()
    return eig


# find the optimal EIG (up to a constant) for D dimensions. we reduce the optimization
# to a univariate problem and use brute-force bisectioning to identify the optimum.
def optimal_eig(alpha, D, sigmasq, S=100 * 1000):
    delta = torch.arange(S + 1).float() / float(S)
    epsilon = 1.0 - delta
    max_eig, optimal_delta = analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq).max(dim=-1)

    for zoom in [1.0e-2, 1.0e-4, 1.0e-6]:
        delta = delta[optimal_delta].item() - zoom + 2.0 * zoom * torch.arange(S + 1).float() / float(S)
        epsilon = 1.0 - delta
        max_eig, optimal_delta = analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq).max(dim=-1)

    return max_eig.item(), delta[optimal_delta].item(), epsilon[optimal_delta].item()


# get the prior precision matrix
def get_prior_precision(alpha, D):
    u = torch.ones(D).unsqueeze(-1)
    u[0:int(D/2), 0] = alpha
    prior_precision = (1.0 + 1.0 / D) * torch.eye(D) - (1.0 / D) * torch.mm(u, u.t())
    return prior_precision


# get the cholesky matrix of the prior precision matrix
def get_prior_scale_tril(alpha, D):
    precision = get_prior_precision(alpha, D)
    return precision.cholesky()


# the total budget in D dimensions
def total_budget(D):
    return 0.5 * D


# produced a normalized design from an unconstrained design
def normalized_design(design, D):
    return total_budget(D) * softmax(design, dim=-1)


# specify the model for given D/alpha/sigma. if learn_design==True,
# the design is a pyro parameter that gets optimized; otherwise it's a fixed constant.
def make_model(D=4, alpha=0.5, sigma=1.0, learn_design=True):
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

    def model_fixed_design(design):
        batch_shape = design.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(batch_shape + (D,)),
                                                         precision_matrix=get_prior_precision(alpha, D)))
            return pyro.sample("y", dist.Normal(x * design, sigma * design.sqrt()).to_event(1))

    if learn_design:
        return model_learn_design
    else:
        return model_fixed_design


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
        est_loss_history.append(loss.detach().clone())
        xi_history.append(pyro.param('budget').detach().clone())
        optim(params)
        optim.step()

    xi_history.append(pyro.param('budget').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def main(num_steps, experiment_name, estimator, seed, start_lr, end_lr):
    pyro.clear_param_store()

    experiment_name = "{}".format(datetime.datetime.now().isoformat())
    results_file = experiment_name + '.pickle'

    alpha = 0.1
    D = 32
    sigma = 1.0
    sigmasq = sigma ** 2

    design_history = []
    t0 = time.time()

    if 'bo' not in estimator:
        model = make_model(D=D, alpha=alpha, sigma=sigma, learn_design=True)
    else:
        model = make_model(D=D, alpha=alpha, sigma=sigma, learn_design=False)

    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)

    if estimator == 'posterior':
        guide = make_posterior_guide(1, alpha, D)
        loss = _differentiable_posterior_loss(model, guide, "y", "x")

    elif estimator == 'nce':
        eig_loss = lambda d, N, **kwargs: nce_eig(model=model, design=d, observation_labels="y",
                                                  target_labels="x", N=N, M=10, **kwargs)
        loss = neg_loss(eig_loss)

    elif estimator == 'ace':
        guide = make_posterior_guide(1, alpha, D)
        eig_loss = _ace_eig_loss(model, guide, 10, "y", "x")
        loss = neg_loss(eig_loss)

    elif 'bo' in estimator:
        if estimator == 'nce.bo':
            assert False, "This branch no bueno"
            eig = lambda d, N, **kwargs: nce_eig(model=model,
                                                 design=normalized_design(d, D),
                                                 observation_labels="y",
                                                 target_labels="x", N=N, M=30, **kwargs)[1][0, ...]
        elif estimator == 'exact.bo':
            eig = lambda d, N, **kwargs: analytic_eig_budget(alpha, D, normalized_design(d, D), sigmasq)[0, ...]

        num_acquisition = 10
        num_parallel = 3
        N_outer = 10000
        lengthscale, noise = torch.tensor(0.3), torch.tensor(0.0001)

        design = 0.3 * torch.randn(1, num_acquisition, D)
        y = eig(design, N_outer).detach().clone()
        design_history.append(normalized_design(design, D))

        kernel = gp.kernels.Matern52(input_dim=D, lengthscale=lengthscale, variance=y.var(unbiased=True))
        num_bo_steps = 35

        for i in range(num_bo_steps):
            Kff = kernel(design)
            Kff += noise * torch.eye(Kff.shape[-1])
            Lff = Kff.cholesky(upper=False)
            new_design = 0.3 * torch.randn(num_parallel, num_acquisition, D)
            new_design.requires_grad_(True)
            minimizer = torch.optim.LBFGS([new_design], max_eval=20)

            def gp_ucb1():
                minimizer.zero_grad()
                KXXnew = kernel(design, new_design)
                LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
                Liy = torch.triangular_solve(y.unsqueeze(-1), Lff, upper=False)[0]
                mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
                KXnewXnew = kernel(new_design)
                var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
                ucb = -(mean + 2*var.sqrt())
                loss = ucb.sum()
                torch.autograd.backward(new_design,
                                        torch.autograd.grad(loss, new_design, retain_graph=True))
                return loss

            minimizer.step(gp_ucb1)
            #final_eig = analytic_eig_budget(alpha, D, normalized_design(new_design, D), sigmasq)
            #print("final eig", final_eig.max().item(), "\n", final_eig)
            new_design = new_design.reshape(-1, D).unsqueeze(0)
            new_y = eig(new_design, N_outer).detach().clone()
            #print("new_y", new_y.shape, "new_design", new_design.shape)
            y_max, which = torch.max(new_y, dim=-1)
            max_eig = analytic_eig_budget(alpha, D, normalized_design(new_design, D), sigmasq)[0, which]
            #print("max_eig", max_eig)
            design = torch.cat([design, new_design], dim=1)
            y = torch.cat([y, new_y])
            #print("y", y.shape, "design", design.shape)

        y_max, which = torch.max(y, dim=-1)
        max_eig = analytic_eig_budget(alpha, D, normalized_design(design, D), sigmasq)[0, which]
        print("\nfinal best eig", max_eig)
        final_design = normalized_design(design, D)[0, which, :]

    else:
        raise ValueError("Unexpected estimator")


    if 'bo' not in estimator:
        gamma = (end_lr/start_lr)**(1/num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, 1, D)  # this is annoying, code needs refactor

        design_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=10,
                                                             num_steps=num_steps, optim=scheduler)

    tf = time.time()
    print("Elapsed time", tf - t0)

    opt_eig, opt_delta, opt_epsilon = optimal_eig(alpha, D, sigmasq)
    opt_design = opt_delta * torch.ones(D)
    opt_design[int(D/2):] = opt_epsilon

    #results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
    #           'design_history': design_history}

    if 'bo' not in estimator:
        initial_design, final_design = design_history[0], design_history[-1]
        initial_design *= total_budget(D)
        final_design *= total_budget(D)
    else:
        initial_design = final_design

    initial_eig = analytic_eig_budget(alpha, D, initial_design, sigmasq)
    flat_eig = analytic_eig_budget(alpha, D, total_budget(D) * torch.ones(D) / D, sigmasq)
    final_eig = analytic_eig_budget(alpha, D, final_design, sigmasq)
    eig_normalizer = opt_eig - flat_eig

    print("Initial/Flat/Final/Optimal EIG:  %.5f / %.5f /  %.5f / %.5f" % (initial_eig, flat_eig, final_eig, opt_eig))
    #print("Initial design:\n", initial_design.data.numpy())
    print("Final design:\n", final_design.data.numpy())
    print("Optimal design:\n", opt_design.data.numpy())
    print("Mean Absolute EIG Error: %.5f" % math.fabs(final_eig - opt_eig))
    print("Normalized Mean Absolute EIG Error: %.5f" % (math.fabs(final_eig - opt_eig) / eig_normalizer))
    print("Mean Absolute Design Error: %.5f" % (final_design - opt_design).abs().sum())

    #with open(results_file, 'wb') as f:
        #    pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=4000, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="ace", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.1, type=float)
    parser.add_argument("--end-lr", default=0.0001, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
