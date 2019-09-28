import argparse
import datetime
import math
import subprocess
import pickle
from functools import lru_cache
import time

import torch
from torch.distributions import constraints
from torch.distributions import transform_to

import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.contrib.util import rmv

from death_process_rb import semi_analytic_eig


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


N = 10
design_dim = 2
prior_mean = torch.tensor(0.)
prior_sd = torch.tensor(1.0)


@lru_cache(5)
def make_y_space(n):
    space = []
    for i in range(n+1):
        for j in range(n-i+1):
            space.append([i, j])
    return torch.tensor(space, dtype=torch.float)


def gp_opt_w_history(loss_fn, num_steps, num_parallel, num_acquisition, lengthscale):

    est_loss_history = []
    xi_history = []
    t = time.time()
    wall_times = []
    X = .01 + 4.99 * torch.rand((num_parallel, num_acquisition, design_dim))

    y = loss_fn(X)

    # GPBO
    y = y.detach().clone()
    kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale),
                                 variance=torch.tensor(.25))
    constraint = torch.distributions.constraints.interval(1e-6, 5.)
    noise = torch.tensor(1e-4)

    def acquire(X, y, sigma, nacq):
        Kff = kernel(X)
        Kff += noise * torch.eye(Kff.shape[-1])
        print(Kff[0, ...])
        Lff = Kff.cholesky(upper=False)
        Xinit = .01 + 4.99 * torch.rand((num_parallel, nacq, design_dim))
        unconstrained_Xnew = transform_to(constraint).inv(Xinit).detach().clone().requires_grad_(True)
        minimizer = torch.optim.LBFGS([unconstrained_Xnew], max_eval=25)

        def gp_ucb1():
            minimizer.zero_grad()
            Xnew = transform_to(constraint)(unconstrained_Xnew)
            # Xnew.register_hook(lambda x: print('Xnew grad', x))
            KXXnew = kernel(X, Xnew)
            LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
            Liy = torch.triangular_solve(y.unsqueeze(-1), Lff, upper=False)[0]
            mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
            KXnewXnew = kernel(Xnew)
            var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
            ucb = -(mean + sigma * var.clamp(min=0.).sqrt())
            loss = ucb.sum()
            torch.autograd.backward(unconstrained_Xnew,
                                    torch.autograd.grad(loss, unconstrained_Xnew, retain_graph=True))
            return loss

        minimizer.step(gp_ucb1)
        X_acquire = transform_to(constraint)(unconstrained_Xnew).detach().clone()
        y_acquire = loss_fn(X_acquire).detach().clone()

        return X_acquire, y_acquire

    for i in range(num_steps):
        X_acquire, y_acquire = acquire(X, y, 2, num_acquisition)

        X = torch.cat([X, X_acquire], dim=-2)
        y = torch.cat([y, y_acquire], dim=-1)

        if i % 10 == 0:
            X_star, y_star = acquire(X, y, 0, 1)
            X_star, y_star = X_star.squeeze(-2), y_star.squeeze(-1)
            print(X_star[0, ...])

            est_loss_history.append(y_star)
            xi_history.append(X_star)
            wall_times.append(time.time() - t)

    X_star, y_star = acquire(X, y, 0, 1)
    X_star, y_star = X_star.squeeze(-2), y_star.squeeze(-1)
    xi_history.append(X_star.detach().clone())
    wall_times.append(time.time() - t)

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    wall_times = torch.tensor(wall_times)

    return xi_history, est_loss_history, wall_times


def main(experiment_name, seed, num_parallel, num_steps, num_acquisition, num_samples):
    output_dir = "./run_outputs/gradinfo/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'

    pyro.clear_param_store()
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)

    # Fix correct loss
    loss = lambda X: semi_analytic_eig(X, prior_mean, prior_sd, n_samples=num_samples)

    xi_history, est_loss_history, wall_times = gp_opt_w_history(
        loss, num_steps, num_parallel, num_acquisition, 1.)

    eig_history = []
    for i in range(xi_history.shape[0] - 1):
        eig_history.append(semi_analytic_eig(xi_history[i, ...], prior_mean, prior_sd, n_samples=20000))
    eig_history.append(semi_analytic_eig(xi_history[-1, ...], prior_mean, prior_sd, n_samples=200000))
    eig_history = torch.stack(eig_history)

    results = {'estimator': 'bo', 'git-hash': get_git_revision_hash(), 'seed': seed,
               'xi_history': xi_history, 'est_eig_history': est_loss_history, 'eig_history': eig_history,
               'wall_times': wall_times}

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BO design optimization for Death Process")
    parser.add_argument("--num-steps", default=200, type=int)
    parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-acquisition", default=5, type=int)
    parser.add_argument("--num-samples", default=100, type=int)
    args = parser.parse_args()
    main(args.name, args.seed, args.num_parallel, args.num_steps, args.num_acquisition, args.num_samples)
