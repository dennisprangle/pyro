import argparse
import datetime
import math
import pickle
import subprocess
import time

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro import poutine
from pyro.contrib.oed.eig import _eig_from_ape, pce_eig, _ace_eig_loss, _posterior_loss
from pyro.contrib.util import rmv
from pyro.util import is_bad


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_pk_model(theta_loc, theta_scale, xi_init, observation_label="y", sd=0.1):
    def pk_model(design_prototype):
        design = pyro.param("xi", xi_init).expand(design_prototype.shape)
        batch_shape = design.shape[:-1]
        with pyro.plate_stack("plate_stack", batch_shape):
            theta = pyro.sample("theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta1, theta2, theta3 = torch.unbind(theta, -1)
            theta1 = theta1.unsqueeze(-1)
            theta2 = theta2.unsqueeze(-1)
            theta3 = theta3.unsqueeze(-1)
            x = 400. * theta2 * (torch.exp(-theta1*design) - torch.exp(-theta2*design)) / (theta3*(theta2-theta1))
            y = pyro.sample(observation_label, dist.Normal(x, sd).to_event(1))
            return y

    return pk_model


class TensorLinear(nn.Module):
    __constants__ = ['bias']

    def __init__(self, *shape, bias=True):
        super(TensorLinear, self).__init__()
        self.in_features = shape[-2]
        self.out_features = shape[-1]
        self.batch_dims = shape[:-2]
        self.weight = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return rmv(self.weight, input) + self.bias


class PosteriorGuide(nn.Module):
    def __init__(self, y_dim, batching):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = TensorLinear(*batching, y_dim, n_hidden)
        self.linear2 = TensorLinear(*batching, n_hidden, n_hidden)
        self.output_layer = TensorLinear(*batching, n_hidden, 9)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        y = y_dict["y"]
        x = self.relu(self.linear1(y))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        posterior_mean = final[..., 0:3]
        posterior_scale_tril_raw = final[..., 3:]
        ## Next lines based on `fill_triangular` from tensorflow probability
        chol_list = [posterior_scale_tril_raw[..., 3:],
                  torch.flip(posterior_scale_tril_raw, [-1])]
        covariance_shape = final.shape[:-1] + (3,3)
        chol = torch.reshape(torch.cat(chol_list, -1), covariance_shape)
        chol = torch.tril(chol)
        chol.diagonal().exp_()
        pyro.module("posterior_guide", self)
        posterior_scale_tril = pyro.param(
            "posterior_scale_tril",
            chol,
            constraint=constraints.lower_cholesky
        )
        batch_shape = design_prototype.shape[:-1]
        with pyro.plate_stack("guide_plate_stack", batch_shape):
            pyro.sample("theta", dist.MultivariateNormal(posterior_mean, scale_tril=posterior_scale_tril))


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))

    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim, time_budget):
    params = None
    est_loss_history = []
    xi_history = []
    baseline = 0.
    t = time.time()
    wall_times = []
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
        if step % 500 == 0:
            xi_history.append(pyro.param('xi').detach().clone())
            est_loss_history.append(loss.detach())
            wall_times.append(time.time() - t)
        optim(params)
        optim.step()
        if step % 500 == 0:
            print(pyro.param("xi")[0, ...])
            print(step)
            print('eig', baseline.squeeze())
        if time_budget and time.time() - t > time_budget:
            break

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    wall_times = torch.tensor(wall_times)

    return xi_history, est_loss_history, wall_times


def main(num_steps, num_samples, time_budget, experiment_name, estimators, seed, num_parallel, start_lr, end_lr,
         device, n, scale):
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

        xi_init = 24. * torch.rand((num_parallel, n), device=device)
        # prior params
        theta_prior_loc = torch.tensor((0.1, 1., 20.), dtype=torch.float32, device=device).log()
        theta_prior_scale = torch.tensor((0.05, 0.05, 0.05), dtype=torch.float32, device=device).sqrt()

        model_learn_xi = make_pk_model(
            theta_prior_loc, theta_prior_scale, xi_init)

        contrastive_samples = num_samples

        # Fix correct loss
        targets = ["theta"]
        if estimator == 'posterior':
            guide = PosteriorGuide(n, (num_parallel,)).to(device)
            loss = _posterior_loss(model_learn_xi, guide, ["y"], targets)

        elif estimator == 'pce':
            eig_loss = lambda d, N, **kwargs: pce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
                N=N, M=contrastive_samples, **kwargs)
            loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            guide = PosteriorGuide(n, (num_parallel,)).to(device)
            eig_loss = _ace_eig_loss(model_learn_xi, guide, contrastive_samples, ["y"], targets)
            loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(num_parallel, n, device=device)  # this is annoying, code needs refactor
        design_prototype = 24. * torch.rand(num_parallel, n, device=device)

        xi_history, est_loss_history, wall_times = opt_eig_loss_w_history(
            design_prototype, loss, num_samples=num_samples, num_steps=num_steps, optim=scheduler,
            time_budget=time_budget)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, targets, est_loss_history, True, {})

        elif estimator in ['pce', 'ace']:
            est_eig_history = -est_loss_history
        else:
            est_eig_history = est_loss_history

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history.cpu(), 'est_eig_history': est_eig_history.cpu(),
                   'wall_times': wall_times.cpu()}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a pharmacokinetic model")
    parser.add_argument("--num-steps", default=500000, type=int)
    parser.add_argument("--time-budget", default=1200, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.001, type=float)
    parser.add_argument("--end-lr", default=0.001, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-n", default=15, type=int)
    parser.add_argument("--scale", default=1., type=float)
    args = parser.parse_args()
    main(args.num_steps, args.num_samples, args.time_budget, args.name, args.estimator, args.seed, args.num_parallel,
         args.start_lr, args.end_lr, args.device, args.n, args.scale)
