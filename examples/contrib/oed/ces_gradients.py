import torch
from torch.distributions import constraints
from torch import nn
import argparse
import subprocess
import datetime
import pickle
from contextlib import ExitStack

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, rexpand, rmv
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss, differentiable_nce_eig, _differentiable_ace_eig_loss
from pyro import poutine
from pyro.contrib.oed.eig import _eig_from_ape
from pyro.util import is_bad


# TODO read from torch float spec
epsilon = torch.tensor(2**-24)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma, observation_sd, observation_label="y",
                   xi_init=torch.ones(6)):
    def ces_model(design_prototype):
        design = pyro.param("xi", xi_init, constraint=constraints.interval(1e-6, 100)).expand(design_prototype.shape)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            rho_shape = batch_shape + (rho_concentration.shape[-1],)
            rho = 0.01 + 0.99 * pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape))).select(-1, 0)
            alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
            alpha = pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
            slope = pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape), slope_sigma.expand(batch_shape)))
            rho, slope = rexpand(rho, design.shape[-2]), rexpand(slope, design.shape[-2])
            d1, d2 = design[..., 0:3], design[..., 3:6]
            U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
            U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
            mean = slope * (U1rho - U2rho)
            sd = slope * observation_sd * (1 + torch.norm(d1 - d2, dim=-1, p=2))
            emission_dist = dist.CensoredSigmoidNormal(mean, sd, 1 - epsilon, epsilon).to_event(1)
            y = pyro.sample(observation_label, emission_dist)
            return y

    return ces_model


class PosteriorGuide(nn.Module):
    def __init__(self):
        super(PosteriorGuide, self).__init__()
        self.linear1 = nn.Linear(1, 256)
        self.linear2 = nn.Linear(256, 256)
        self.rho_concentration = nn.Linear(256, 2)
        self.alpha_concentration = nn.Linear(256, 3)
        self.slope_mu = nn.Linear(256, 1)
        self.slope_sigma = nn.Linear(256, 1)

        self.softplus = nn.Softplus()

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        y = y_dict["y"]
        y, y1m = y.clamp(1e-35, 1), (1. - y).clamp(1e-35, 1)
        s = y.log() - y1m.log()
        x = self.softplus(self.linear1(s))
        x = self.softplus(self.linear2(x))
        rho_concentration = 1e-6 + self.softplus(self.rho_concentration(x))
        alpha_concentration = 1e-6 + self.softplus(self.alpha_concentration(x))
        slope_mu = self.slope_mu(x).squeeze(-1)
        slope_sigma = 1e-6 + self.softplus(self.slope_sigma(x)).squeeze(-1)

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            rho_shape = batch_shape + (rho_concentration.shape[-1],)
            pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape)))
            alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
            pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
            pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape), slope_sigma.expand(batch_shape)))


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

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def main(num_steps, experiment_name, estimators, seed, start_lr, end_lr):
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

        xi_init = 0.01 + 99.99 * torch.rand(6)
        observation_sd = torch.tensor(.005)
        # Change the prior distribution here
        #model_learn_xi = make_ces_model(torch.ones(1, 2), torch.ones(1, 3),
        #                                torch.ones(1), .001*torch.ones(1), observation_sd, xi_init=xi_init)
        model_learn_xi = make_ces_model(torch.tensor([[0.5, 10., 0.1]]), torch.tensor([[0.1, 0.2, 12.]]),
                                                     torch.tensor([1.5]), torch.tensor([1.5]), observation_sd, xi_init=xi_init)

        # Fix correct loss
        if estimator == 'posterior':
            guide = PosteriorGuide()
            loss = _differentiable_posterior_loss(model_learn_xi, guide, ["y"], ["rho", "alpha", "slope"])

        elif estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: differentiable_nce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=["rho", "alpha", "slope"],
                N=N, M=100, **kwargs)
            loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            guide = PosteriorGuide()
            eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, 10, ["y"], ["rho", "alpha", "slope"])
            loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr/start_lr)**(1/num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, 1, 6)  # this is annoying, code needs refactor

        xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=10,
                                                              num_steps=num_steps, optim=scheduler)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, ["y"], est_loss_history, True, {})
        else:
            est_eig_history = -est_loss_history
        # eig_history = semi_analytic_eig(xi_history, torch.tensor(0.), torch.tensor(0.25))

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=2000, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.01, type=float)
    parser.add_argument("--end-lr", default=0.0005, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
