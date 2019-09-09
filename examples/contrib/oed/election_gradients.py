import argparse
import datetime
from contextlib import ExitStack
import math
import subprocess
import pickle
from functools import lru_cache
import pandas as pd

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.oed.eig import _eig_from_ape
from pyro.contrib.oed.eig import _posterior_loss
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss
from pyro.contrib.util import rmv, iter_plates_to_shape, lexpand, rvv, rexpand


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def prepare_prior_from_data():
    electoral_college_votes = pd.read_pickle("examples/contrib/oed/electoral_college_votes.pickle")
    ec_votes_tensor = torch.tensor(electoral_college_votes.values, dtype=torch.float).squeeze()
    frame = pd.read_pickle("examples/contrib/oed/us_presidential_election_data_historical.pickle")
    results_2012 = torch.tensor(frame[2012].values, dtype=torch.float)
    prior_mean = torch.log(results_2012[..., 0] / results_2012[..., 1])
    idx = 2 * torch.arange(10)
    as_tensor = torch.tensor(frame.values, dtype=torch.float)
    logits = torch.log(as_tensor[..., idx] / as_tensor[..., idx + 1]).transpose(0, 1)
    mean = logits.mean(0)
    sample_covariance = (1 / (logits.shape[0] - 1)) * (
            (logits.unsqueeze(-1) - mean) * (logits.unsqueeze(-2) - mean)
    ).sum(0)
    prior_covariance = sample_covariance + 0.01 * torch.eye(sample_covariance.shape[0])

    return prior_mean, prior_covariance, ec_votes_tensor, frame


xi_init = torch.ones(51)


def make_model(prior_mean, prior_covariance, ec_votes_tensor):
    def model(design_prototype):
        polling_allocation = pyro.param('xi', xi_init, constraint=constraints.positive)
        polling_allocation = polling_allocation * 1000 / polling_allocation.sum(-1)
        polling_allocation = polling_allocation.expand(design_prototype.shape)

        with ExitStack() as stack:
            for plate in iter_plates_to_shape(polling_allocation.shape[:-1]):
                stack.enter_context(plate)
            theta = pyro.sample("theta", dist.MultivariateNormal(prior_mean, covariance_matrix=prior_covariance))
            poll_results = pyro.sample("y", dist.Binomial(polling_allocation, logits=theta).to_event(1))
            dem_win_state = (theta > 0.).float()
            dem_electoral_college_votes = ec_votes_tensor * dem_win_state
            dem_win = (dem_electoral_college_votes.sum(-1) / ec_votes_tensor.sum(-1) > .5).float()
            pyro.sample("w", dist.Delta(dem_win))
            return poll_results, dem_win, theta

    return model


class OutcomePredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(51, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 1)

    def compute_dem_probability(self, y):
        y = nn.functional.relu(self.lin1(y))
        y = nn.functional.relu(self.lin2(y))
        return self.lin3(y)

    def forward(self, y_dict, design, observation_labels, target_labels):
        pyro.module("posterior_guide", self)

        y = y_dict["y"]
        dem_prob = self.compute_dem_probability(y).squeeze()
        pyro.sample("w", dist.Bernoulli(logits=dem_prob))


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
        baseline = loss.detach()
        xi_history.append(pyro.param('xi').detach().clone())
        optim(params)
        optim.step()
        print(pyro.param("xi"))

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def main(num_steps, experiment_name, estimators, seed, start_lr, end_lr):
    # torch.set_printoptions(precision=10)
    # design = torch.tensor([[0.9879, 1.8558], [0.8112, 1.4376], [1.0836, 1.5458], [0.96, 1.58]])
    # eig = semi_analytic_eig(design, torch.tensor(0.), torch.tensor(0.25), n_samples=1000000)
    # print(-eig + eig[-1, ...])
    # print((design - torch.tensor([0.96, 1.58])).pow(2).sum(-1).sqrt())
    # raise
    output_dir = "./run_outputs/gradinfo/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'
    estimators = estimators.split(",")

    prior_mean, prior_covariance, ec_votes_tensor, frame = prepare_prior_from_data()
    model = make_model(prior_mean, prior_covariance, ec_votes_tensor)

    for estimator in estimators:
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)

        # Fix correct loss
        if estimator == 'posterior':
            guide = OutcomePredictor()
            loss = _differentiable_posterior_loss(model, guide, ["y"], ["w"])

        # elif estimator == 'nce':
        #     eig_loss = lambda d, N, **kwargs: differentiable_nce_eig(
        #         model=model_learn_xi, design=d, observation_labels=["i1", "i2"], target_labels=["b"], N=N, M=10,
        #         **kwargs)
        #     loss = neg_loss(eig_loss)
        #
        # elif estimator == 'ace':
        #     guide = PosteriorGuide()
        #     eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, 10, ["i1", "i2"], ["b"])
        #     loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(51)  # this is annoying, code needs refactor

        xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=10,
                                                              num_steps=num_steps, optim=scheduler)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model, design_prototype, ["b"], est_loss_history, True, {})
        else:
            est_eig_history = -est_loss_history

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history}

        final_allocation = pd.DataFrame({"Sample size": 1000 * (pyro.param("xi") / pyro.param("xi").sum(-1)).detach().numpy(),
                                         "State": frame.index}).set_index("State")
        print(final_allocation)

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=2000, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.001, type=float)
    parser.add_argument("--end-lr", default=0.00001, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
