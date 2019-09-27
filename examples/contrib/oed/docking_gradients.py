import torch
from torch.distributions import constraints
import torch.nn.functional as F
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
from pyro.contrib.util import iter_plates_to_shape, lexpand, rmv
from pyro.contrib.oed.differentiable_eig import (
        _differentiable_posterior_loss, differentiable_nce_eig, _differentiable_ace_eig_loss,
        differentiable_nce_proposal_eig, _saddle_marginal_loss
        )
from pyro import poutine
from pyro.contrib.oed.eig import _eig_from_ape, nce_eig, _ace_eig_loss, nmc_eig, vnmc_eig
from pyro.util import is_bad


# TODO read from torch float spec
epsilon = torch.tensor(2**-22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def invsoftplus(x):
    return (x.exp() - 1).log()


def invsigmoid(y):
    y, y1m = y.clamp(1e-35, 1), (1. - y).clamp(1e-35, 1)
    return y.log() - y1m.log()


def sigmoid(x, top, bottom, ee50, slope):
    return (top - bottom) * torch.sigmoid((x - ee50) * slope) + bottom


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


def make_posterior_guide(d, top_prior_c, bottom_prior_c, ee50_prior_mu, ee50_prior_sigma, slope_prior_mu,
                         slope_prior_sigma):
    def posterior_guide(y_dict, design, observation_labels, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)

        top_mult = pyro.param("A_top", torch.zeros(*d, y.shape[-1]))
        top_confidence = pyro.param("top_v", top_prior_c.sum(), constraint=constraints.positive)
        top_c = top_confidence * torch.sigmoid(
            invsigmoid(top_prior_c[..., 0] / top_prior_c.sum(-1)) + (top_mult * (y - .5)).sum(-1)
        )
        top_c = torch.stack([top_c, top_confidence - top_c], dim=-1)

        bottom_mult = pyro.param("A_ bottom", torch.ones(*d, y.shape[-1]))
        bottom_confidence = pyro.param("bottom_v", bottom_prior_c.sum(), constraint=constraints.positive)
        bottom_c = bottom_confidence * torch.sigmoid(
            invsigmoid(bottom_prior_c[..., 0] / bottom_prior_c.sum(-1)) + (bottom_mult * (y - .5)).sum(-1)
        )
        bottom_c = torch.stack([bottom_c, bottom_confidence - bottom_c], dim=-1)

        ee50_mult = pyro.param("A_ee50", torch.zeros(*d, y.shape[-1]))
        ee50_mu = ee50_prior_mu + (ee50_mult * y).sum(-1)
        ee50_sigma = pyro.param("ee50_sd", ee50_prior_sigma, constraint=constraints.positive)

        slope_mult = pyro.param("A_slope", torch.zeros(*d, y.shape[-1]))
        slope_mu = slope_prior_mu + (slope_mult * y).sum(-1)
        slope_sigma = pyro.param("slope_sd", slope_prior_sigma, constraint=constraints.positive)

        print(ee50_mu)

        batch_shape = design.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            pyro.sample("top", dist.Dirichlet(top_c))
            pyro.sample("bottom", dist.Dirichlet(bottom_c))
            pyro.sample("ee50", dist.Normal(ee50_mu, ee50_sigma))
            pyro.sample("slope", dist.Normal(slope_mu, slope_sigma))

    return posterior_guide


class PosteriorGuide(nn.Module):
    def __init__(self, y_dim):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = nn.Linear(y_dim, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 2 + 2 + 2 + 2)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def set_prior(self, rho_concentration, alpha_concentration, slope_mu, slope_sigma):
        self.prior_rho_concentration = rho_concentration
        self.prior_alpha_concentration = alpha_concentration
        self.prior_slope_mu = slope_mu
        self.prior_slope_sigma = slope_sigma

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        y = y_dict["y"] - .5
        x = self.relu(self.linear1(y))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        top_c = self.softplus(final[..., 0:2])
        bottom_c = self.softplus(final[..., 2:4])
        ee50_mu = final[..., 4]
        ee50_sigma = self.softplus(final[..., 5])
        slope_mu = final[..., 6]
        slope_sigma = self.softplus(final[..., 7])

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            pyro.sample("top", dist.Dirichlet(top_c))
            pyro.sample("bottom", dist.Dirichlet(bottom_c))
            pyro.sample("ee50", dist.Normal(ee50_mu, ee50_sigma))
            pyro.sample("slope", dist.Normal(slope_mu, slope_sigma))


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim, lower, upper, n_high_acc, h_freq):

    params = None
    est_loss_history = []
    lower_history = []
    upper_history = []
    xi_history = []
    baseline = 0.
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, evaluation=True, control_variate=baseline)
        baseline = -loss.detach()
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward(retain_graph=True)
        est_loss_history.append(loss.detach())
        optim(params)
        optim.step()
        print(pyro.param("xi").squeeze())
        print('eig', baseline.squeeze())

        if step % h_freq == 0:
            _, low = lower(design, n_high_acc, evaluation=True)
            _, up = upper(design, n_high_acc, evaluation=True)
            lower_history.append(low.detach())
            upper_history.append(up.detach())

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history, lower_history, upper_history


def main(num_steps, high_acc_freq, num_samples, experiment_name, estimators, seed, start_lr, end_lr):
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

        D = 100
        xi_init = torch.linspace(-70., -30, D)
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

        contrastive_samples = num_samples

        # Fix correct loss
        targets = ["top", "bottom", "ee50", "slope"]
        if estimator == 'posterior':
            m_final = 200
            guide = PosteriorGuide(D)
            loss = _differentiable_posterior_loss(model_learn_xi, guide, ["y"], targets)
            high_acc = loss
            upper_loss = lambda d, N, **kwargs: vnmc_eig(model_learn_xi, d, "y", targets, (N, int(math.sqrt(N))), 0, guide, None)

        elif estimator == 'nce':
            m_final = 400
            eig_loss = lambda d, N, **kwargs: differentiable_nce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
                N=N, M=contrastive_samples, **kwargs)
            loss = neg_loss(eig_loss)
            high_acc = lambda d, N, **kwargs: nce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
                N=N, M=int(math.sqrt(N)), **kwargs)
            upper_loss = lambda d, N, **kwargs: nmc_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
                N=N, M=int(math.sqrt(N)), **kwargs)

        elif estimator == 'ace':
            m_final = 200
            guide = PosteriorGuide(D)
            eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, contrastive_samples, ["y"],
                                                    ["top", "bottom", "ee50", "slope"])
            loss = neg_loss(eig_loss)
            high_acc = _ace_eig_loss(model_learn_xi, guide, m_final, "y", targets)
            upper_loss = lambda d, N, **kwargs: vnmc_eig(model_learn_xi, d, "y", targets, (N, int(math.sqrt(N))), 0, guide, None)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, D)  # this is annoying, code needs refactor

        xi_history, est_loss_history, lower_history, upper_history = opt_eig_loss_w_history(
            design_prototype, loss, num_samples=num_samples, num_steps=num_steps, optim=scheduler, lower=high_acc,
            upper=upper_loss, n_high_acc=m_final**2, h_freq=high_acc_freq)

        # m_final = 200
        # if estimator == 'nce':
        #     final_lower = nce_eig(model_learn_xi, design_prototype, "y", ["top", "bottom", "ee50", "slope"], N=m_final**2, M=m_final)
        #     final_upper = nmc_eig(model_learn_xi, design_prototype, "y",  ["top", "bottom", "ee50", "slope"], N=m_final**2, M=m_final)
        # elif estimator == 'ace':
        #     ls = _ace_eig_loss(model_learn_xi, guide, m_final, "y", ["top", "bottom", "ee50", "slope"])
        #     final_lower = ls(design_prototype, m_final**2)
        #     final_upper = vnmc_eig(model_learn_xi, design_prototype, "y", ["top", "bottom", "ee50", "slope"], (m_final**2, m_final), 0, guide, None)


        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, targets, est_loss_history, True, {})
            lower_history = _eig_from_ape(model_learn_xi, design_prototype, targets, lower_history, True, {})

        elif estimator in ['nce', 'nce-proposal', 'ace']:
            est_eig_history = -est_loss_history
        else:
            est_eig_history = est_loss_history

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history,
                   'lower_history': lower_history, 'upper_history': upper_history}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=2000, type=int)
    parser.add_argument("--high-acc-freq", default=1000, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.001, type=float)
    parser.add_argument("--end-lr", default=0.0001, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.high_acc_freq, args.num_samples, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
