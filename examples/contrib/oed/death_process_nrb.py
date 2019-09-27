import argparse
import datetime
from contextlib import ExitStack
import math
import subprocess
import pickle
from functools import lru_cache

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.oed.eig import _eig_from_ape
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss, differentiable_nce_eig, _differentiable_ace_eig_loss
from pyro.contrib.util import rmv, iter_plates_to_shape, lexpand, rvv, rexpand


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


N = 10
xi_init = 0.1*torch.ones(2)
prior_mean = torch.tensor(0.)
prior_sd = torch.tensor(1.0)


@lru_cache(5)
def make_y_space(n):
    space = []
    for i in range(n+1):
        for j in range(n-i+1):
            space.append([i, j])
    return torch.tensor(space, dtype=torch.float)


def semi_analytic_eig(design, prior_mean, prior_sd, n_samples=1000):
    with pyro.plate("plate0", n_samples):
        samples = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))

    lp1m1 = -(samples * design[..., [0]]).unsqueeze(-1)
    lp2m1 = -(samples * design[..., [1]]).unsqueeze(-1)

    def log_prob(lp1m1, lp2m1):
        lp1 = (1 - lp1m1.exp()).log()
        lp2 = (1 - lp2m1.exp()).log()

        y = make_y_space(N)
        log_prob_y = torch.lgamma(torch.tensor(N + 1, dtype=torch.float)) - torch.lgamma(y[:, 0] + 1) - torch.lgamma(y[:, 1] + 1) \
                     - torch.lgamma(N - y.sum(-1) + 1) + y[:, 0] * lp1 + y[:, 1] * lp2 + (N - y[:, 0]) * lp1m1 \
                     + (N - y[:, 0] - y[:, 1]) * lp2m1
        return log_prob_y

    likelihoods = log_prob(lp1m1, lp2m1)
    marginal = likelihoods.logsumexp(-2, keepdim=True) - math.log(n_samples)
    kls = (likelihoods.exp() * (likelihoods - marginal)).sum(-1)
    return kls.mean(-1)


def model_learn_xi(design_prototype):
    xi = pyro.param('xi', xi_init, constraint=constraints.positive)
    batch_shape = design_prototype.shape
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        b = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))
        p1 = 1 - torch.exp(-b * xi[0])
        infected1 = pyro.sample("i1", dist.Binomial(N, p1))
        p2 = 1 - torch.exp(-b * xi[1])
        infected2 = pyro.sample("i2", dist.Binomial(N - infected1, p2))
        return infected1, infected2


class PosteriorGuide(nn.Module):
    def __init__(self):
        super(PosteriorGuide, self).__init__()
        self.linear1 = nn.Linear(2, 8)
        self.linear2 = nn.Linear(8, 8)
        self.mu = nn.Linear(8, 1)
        self.sigma = nn.Linear(8, 1)
        self.softplus = nn.Softplus()

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        i1, i2 = y_dict["i1"], y_dict["i2"]
        s1, s2 = 1./(1.1 - i1/N), 1./(1.1 - i2/(N + 1 - i1))
        all_inputs = torch.cat([s1, s2], dim=-1)
        x = self.softplus(self.linear1(all_inputs))
        x = self.softplus(self.linear2(x))
        mu = self.mu(x)
        sigma = self.softplus(self.sigma(x))

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            pyro.sample("b", dist.LogNormal(mu.expand(batch_shape), sigma.expand(batch_shape)))


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

        # Fix correct loss
        if estimator == 'posterior':
            guide = PosteriorGuide()
            loss = _differentiable_posterior_loss(model_learn_xi, guide, ["i1", "i2"], ["b"])

        elif estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: differentiable_nce_eig(
                model=model_learn_xi, design=d, observation_labels=["i1", "i2"], target_labels=["b"], N=N, M=10,
                **kwargs)
            loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            guide = PosteriorGuide()
            eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, 10, ["i1", "i2"], ["b"])
            loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr/start_lr)**(1/num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1)  # this is annoying, code needs refactor

        xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=10,
                                                              num_steps=num_steps, optim=scheduler)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, ["b"], est_loss_history, True, {})
        else:
            est_eig_history = -est_loss_history
        eig_history = semi_analytic_eig(xi_history, prior_mean, prior_sd)

        # Build heatmap
        grid_points = 100
        b0low = min(0.05, xi_history[:, 0].min())
        b0up = max(3, xi_history[:, 0].max()) + 0.1
        b1low = min(0.05, xi_history[:, 1].min())
        b1up = max(3, xi_history[:, 1].max()) + 0.1
        xi1 = torch.linspace(b0low, b0up, grid_points)
        xi2 = torch.linspace(b1low, b1up, grid_points)
        d1 = xi1.expand(grid_points, grid_points).unsqueeze(-1)
        d2 = xi2.unsqueeze(-1).expand(grid_points, grid_points).unsqueeze(-1)
        d = torch.cat([d1, d2], dim=-1)
        eig_heatmap = semi_analytic_eig(d, prior_mean, prior_sd)
        extent = [b0low, b0up, b1low, b1up]

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history, 'eig_history': eig_history,
                   'eig_heatmap': eig_heatmap, 'extent': extent}

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
