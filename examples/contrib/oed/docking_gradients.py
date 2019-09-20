import torch
from torch.distributions import constraints
from torch import nn
import argparse
import math
import subprocess
import datetime
import pickle
import logging
from contextlib import ExitStack

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, rexpand, rmv
from pyro.contrib.oed.differentiable_eig import (
        _differentiable_posterior_loss, differentiable_nce_eig, _differentiable_ace_eig_loss,
        differentiable_nce_proposal_eig, _saddle_marginal_loss
        )
from pyro import poutine
from pyro.contrib.oed.eig import _eig_from_ape
from pyro.util import is_bad


# TODO read from torch float spec
epsilon = torch.tensor(2**-22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def sigmoid(x, top, bottom, ee50, slope):
    return (top - bottom) * nn.functional.sigmoid((x - ee50) * slope) + bottom


def make_docking_model(top_c, bottom_c, ee50_mu, ee50_sigma, slope_mu, slope_sigma, observation_label="y",
                       xi_init=torch.ones(6)):
    def docking_model(design_prototype):
        design = pyro.param("xi", xi_init, constraint=constraints.interval(-100, -1e-6)).expand(design_prototype.shape)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            top = pyro.sample("top", dist.Dirichlet(top_c)).select(-1, 0).unsqueeze(-1)
            bottom = pyro.sample("bottom", dist.Dirichlet(bottom_c)).select(-1, 0).unsqueeze(-1)
            ee50 = pyro.sample("ee50", dist.Normal(ee50_mu, ee50_sigma)).unsqueeze(-1)
            slope = pyro.sample("slope", dist.Normal(slope_mu, slope_sigma)).unsqueeze(-1)
            hit_rate = sigmoid(design, top, bottom, ee50, slope)
            y = pyro.sample(observation_label, dist.Bernoulli(hit_rate).to_event(1))
            return y

    return docking_model


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim):

    params = None
    est_loss_history = []
    xi_history = []
    baseline = 0.
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, evaluation=True, control_variate=baseline)
        baseline = loss.detach()
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward(retain_graph=True)
        est_loss_history.append(loss)
        xi_history.append(pyro.param('xi').detach().clone())
        optim(params)
        optim.step()
        print(pyro.param("xi").squeeze())
        print('eig', baseline.squeeze())

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def main(num_steps, num_samples, experiment_name, estimators, seed, start_lr, end_lr):
    output_dir = "./run_outputs/gradinfo/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'
    estimators = estimators.split(",")

    for estimator in estimators:
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)

        D = 20
        xi_init = torch.linspace(-100., -1e-6, D)
        # xi_init = torch.cat([xi_init, xi_init], dim=-1)
        # Change the prior distribution here
        # prior params
        top_prior_concentration = torch.tensor([25., 75.])
        bottom_prior_concentration = torch.tensor([4., 96.])
        ee50_prior_mu, ee50_prior_sd = torch.tensor(-50.), torch.tensor(15.)
        slope_prior_mu, slope_prior_sd = torch.tensor(-0.15), torch.tensor(0.1)
        model_learn_xi = make_docking_model(
            top_prior_concentration, bottom_prior_concentration, ee50_prior_mu, ee50_prior_sd, slope_prior_mu,
            slope_prior_sd, xi_init=xi_init)

        contrastive_samples = num_samples ** 2

        # Fix correct loss
        # if estimator == 'posterior':
        #     guide = PosteriorGuide(tuple())
        #     guide.set_prior(rho_concentration, alpha_concentration, slope_mu, slope_sigma)
        #     loss = _differentiable_posterior_loss(model_learn_xi, guide, ["y"], ["rho", "alpha", "slope"])

        if estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: differentiable_nce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=["rho", "alpha", "slope"],
                N=N, M=contrastive_samples, **kwargs)
            loss = neg_loss(eig_loss)

        # elif estimator == 'nce-proposal':
        #     eig_loss = lambda d, N, **kwargs: differentiable_nce_proposal_eig(
        #             model=model_learn_xi, design=d, observation_labels=["y"], target_labels=['rho', 'alpha', 'slope'],
        #             proposal=proposal, N=N, M=contrastive_samples, **kwargs)
        #     loss = neg_loss(eig_loss)

        # elif estimator == 'ace':
        #     guide = LinearPosteriorGuide(tuple())
        #     guide.set_prior(rho_concentration, alpha_concentration, slope_mu, slope_sigma)
        #     eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, contrastive_samples, ["y"],
        #                                             ["rho", "alpha", "slope"])
        #     loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, D)  # this is annoying, code needs refactor

        xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=num_samples,
                                                              num_steps=num_steps, optim=scheduler)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, ["y"], est_loss_history, True, {})
        elif estimator in ['nce', 'nce-proposal', 'ace']:
            est_eig_history = -est_loss_history
        else:
            est_eig_history = est_loss_history

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=2000, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.01, type=float)
    parser.add_argument("--end-lr", default=0.0005, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.num_samples, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
