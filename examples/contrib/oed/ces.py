import torch
from torch.distributions import transform_to
from torch import nn
import argparse
import subprocess
import datetime
import pickle
import time
import os
from functools import partial
from contextlib import ExitStack

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv
from pyro.contrib.oed.eig import marginal_eig, elbo_learn, nmc_eig
import pyro.contrib.gp as gp
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss
from pyro.contrib.oed.eig import opt_eig_ape_loss
from pyro.util import is_bad


# TODO read from torch float spec
epsilon = torch.tensor(2**-24)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma, observation_sd, observation_label="y"):
    def ces_model(design):
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
            print('rho', rho.max().item(), rho.min().item())
            d1, d2 = design[..., 0:3], design[..., 3:6]
            U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
            U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
            mean = slope * (U1rho - U2rho)
            print('latent samples:', rho.mean().item(), alpha.mean().item(), slope.mean().item(), slope.median().item())
            print('mean', mean.mean().item(), mean.std().item(), mean.min().item(), mean.max().item())
            sd = slope * observation_sd * (1 + torch.norm(d1 - d2, dim=-1, p=2))
            print('sd', sd.mean(), sd.std(), sd.min(), sd.max())
            emission_dist = dist.CensoredSigmoidNormal(mean, sd, 1 - epsilon, epsilon).to_event(1)
            y = pyro.sample(observation_label, emission_dist)
            return y

    return ces_model


def make_learn_xi_model(model, xi_init, constraint):
    def model_learn_xi(design_prototype):
        design = pyro.param("xi", xi_init, constraint=constraint)
        design = design.expand(design_prototype.shape)
        return model(design)
    return model_learn_xi


def elboguide(design, dim=10):
    rho_concentration = pyro.param("rho_concentration", torch.ones(dim, 1, 2),
                                   constraint=torch.distributions.constraints.positive)
    alpha_concentration = pyro.param("alpha_concentration", torch.ones(dim, 1, 3),
                                     constraint=torch.distributions.constraints.positive)
    slope_mu = pyro.param("slope_mu", torch.ones(dim, 1))
    slope_sigma = pyro.param("slope_sigma", 3.*torch.ones(dim, 1),
                             constraint=torch.distributions.constraints.positive)
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)
        rho_shape = batch_shape + (rho_concentration.shape[-1],)
        pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape)))
        alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
        pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
        pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape),
                                            slope_sigma.expand(batch_shape)))


def marginal_guide(mu_init, log_sigma_init, shape, label):
    def guide(design, observation_labels, target_labels):
        mu = pyro.param("marginal_mu", mu_init * torch.ones(*shape))
        log_sigma = pyro.param("marginal_log_sigma", log_sigma_init * torch.ones(*shape))
        ends = pyro.param("marginal_ends", 1./3 * torch.ones(*shape, 3),
                          constraint=torch.distributions.constraints.simplex)
        # print('ends', ends)
        # print('mu', mu)
        # print('log_sigma', log_sigma)
        response_dist = dist.CensoredSigmoidNormalEnds(
            loc=mu, scale=torch.exp(log_sigma), upper_lim=1. - epsilon, lower_lim=epsilon,
            p0=ends[..., 0], p1=ends[..., 1], p2=ends[..., 2]
        ).to_event(1)
        pyro.sample(label, response_dist)
    return guide


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

        print('slope_sigma', slope_sigma)
        #rho_concentration.register_hook(lambda x: print('rhocgrad', x))
        #alpha_concentration.register_hook(lambda x: print('acgrad', x))
        #slope_mu.register_hook(lambda x: print('slopemugrad', x))
        #slope_sigma.register_hook(lambda x: print('slopesigmagrad', x))
        batch_shape = design_prototype.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            rho_shape = batch_shape + (rho_concentration.shape[-1],)
            pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape)))
            alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
            pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
            pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape), slope_sigma.expand(batch_shape)))


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def main(num_steps, num_parallel, experiment_name, typs, seed, lengthscale):
    output_dir = "./run_outputs/ces/"
    if not experiment_name:
        experiment_name = output_dir+"{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir+experiment_name
    results_file = experiment_name + '.result_stream.pickle'
    try:
        os.remove(experiment_name)
    except OSError:
        pass
    typs = typs.split(",")
    observation_sd = torch.tensor(.005)

    for typ in typs:
        print("Type", typ)
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2**30)
            pyro.set_rng_seed(seed)
        marginal_mu_init, marginal_log_sigma_init = 0., 6.
        oed_n_samples, oed_n_steps, oed_final_n_samples, oed_lr = 10, 1250, 2000, [0.1, 0.01, 0.001]
        elbo_n_samples, elbo_n_steps, elbo_lr = 10, 1000, 0.04
        num_acq = 50
        num_bo_steps = 4
        design_dim = 6

        guide = marginal_guide(marginal_mu_init, marginal_log_sigma_init, (num_parallel, num_acq, 1), "y")
        posterior_guide = PosteriorGuide()

        prior = make_ces_model(torch.ones(num_parallel, 1, 2), torch.ones(num_parallel, 1, 3),
                               torch.ones(num_parallel, 1), 3.*torch.ones(num_parallel, 1), observation_sd)
        rho_concentration = torch.ones(num_parallel, 1, 2)
        alpha_concentration = torch.ones(num_parallel, 1, 3)
        slope_mu, slope_sigma = torch.ones(num_parallel, 1), 3.*torch.ones(num_parallel, 1)

        true_model = pyro.condition(make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma,
                                                   observation_sd),
                                    {"rho": torch.tensor([.9, .1]), "alpha": torch.tensor([.2, .3, .5]),
                                     "slope": torch.tensor(10.)})

        d_star_designs = torch.tensor([])
        ys = torch.tensor([])

        for step in range(num_steps):
            print("Step", step)
            model = make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma, observation_sd)
            results = {'typ': typ, 'step': step, 'git-hash': get_git_revision_hash(), 'seed': seed,
                       'lengthscale': lengthscale, 'observation_sd': observation_sd}

            # Design phase
            t = time.time()

            if typ in ['marginal', 'nmc']:
                # Initialization
                noise = torch.tensor(0.2).pow(2)
                # X = 100*rexpand(torch.rand((num_parallel, num_acq)), 4)
                X = .01 + 99.99 * torch.rand((num_parallel, num_acq, 1, design_dim))

                if typ == 'marginal':
                    def f(X):
                        n_steps = oed_n_steps // len(oed_lr)
                        for lr in oed_lr:
                            marginal_eig(model, X, observation_labels=["y"], target_labels=["rho", "alpha", "slope"],
                                         num_samples=oed_n_samples, num_steps=n_steps, guide=guide,
                                         optim=optim.Adam({"lr": lr}))
                        return marginal_eig(model, X, observation_labels=["y"], target_labels=["rho", "alpha", "slope"],
                                            num_samples=oed_n_samples, num_steps=1, guide=guide,
                                            final_num_samples=oed_final_n_samples, optim=optim.Adam({"lr": 1e-6}))
                elif typ == 'nmc':
                    def f(X):
                        return torch.cat([nmc_eig(model, X[:, 25 * i:25 * (i + 1), ...], ["y"],
                                                  ["rho", "alpha", "slope"], N=70*70, M=70)
                                          for i in range(X.shape[1]//25)], dim=1)

                y = f(X)

                # Random search
                # # print(y.mean(1), y.max(1), y.min(1), y.std(1))
                # d_star_index = torch.argmax(y, dim=1)
                # # print(d_star_index.shape)
                # # print(d_star_index)
                # d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2)

                # GPBO
                y = y.detach().clone()
                kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale),
                                             variance=y.var(unbiased=True))
                X = X.squeeze(-2)
                constraint = torch.distributions.constraints.interval(1e-6, 100.)

                for i in range(num_bo_steps):
                    Kff = kernel(X)
                    Kff += noise * torch.eye(Kff.shape[-1])
                    Lff = Kff.cholesky(upper=False)
                    Xinit = .01 + 99.99 * torch.rand((num_parallel, num_acq, design_dim))
                    unconstrained_Xnew = transform_to(constraint).inv(Xinit).detach().clone().requires_grad_(True)
                    minimizer = torch.optim.LBFGS([unconstrained_Xnew], max_eval=20)

                    def gp_ucb1():
                        minimizer.zero_grad()
                        Xnew = transform_to(constraint)(unconstrained_Xnew)
                        # Xnew.register_hook(lambda x: print('Xnew grad', x))
                        KXXnew = kernel(X, Xnew)
                        LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
                        Liy = torch.triangular_solve(y.unsqueeze(-1).clamp(max=20.), Lff, upper=False)[0]
                        mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
                        KXnewXnew = kernel(Xnew)
                        var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
                        ucb = -(mean + 2*var.sqrt())
                        loss = ucb.sum()
                        torch.autograd.backward(unconstrained_Xnew,
                                                torch.autograd.grad(loss, unconstrained_Xnew, retain_graph=True))
                        return loss

                    minimizer.step(gp_ucb1)
                    X_acquire = transform_to(constraint)(unconstrained_Xnew).detach().clone()
                    # print('X_acquire', X_acquire)
                    y_acquire = f(X_acquire.unsqueeze(-2)).detach().clone()
                    # print('y_acquire', y_acquire)

                    X = torch.cat([X, X_acquire], dim=1)
                    y = torch.cat([y, y_acquire], dim=1)

                max_eig, d_star_index = torch.max(y, dim=1)
                print('max EIG', max_eig)
                results['max EIG'] = max_eig
                d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2).unsqueeze(-2)

            elif typ == 'posterior-grad':
                constraint = torch.distributions.constraints.interval(1e-6, 100.)
                xi_init = torch.tensor([1., 2., 3., 4., 5., 1.]).expand((num_parallel, 1, 1, design_dim))
                pyro.param("xi", xi_init, constraint=constraint)
                pyro.get_param_store().replace_param("xi", xi_init, pyro.param("xi"))

                model_learn_xi = make_learn_xi_model(model, xi_init, constraint)
                posterior_guide.apply(weight_reset)

                loss = _differentiable_posterior_loss(model_learn_xi, posterior_guide, ["y"], ["rho", "alpha", "slope"])

                start_lr, end_lr = 0.01, 0.0005
                gamma = (end_lr / start_lr) ** (1 / 7500)
                scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                                      'gamma': gamma})

                design_prototype = torch.zeros(num_parallel, 1, 1, 6)  # this is annoying, code needs refactor

                opt_eig_ape_loss(design_prototype, loss, num_samples=10, num_steps=7500, optim=scheduler)

                d_star_design = pyro.param("xi").detach().clone()

            elif typ == 'rand':
                d_star_design = .01 + 99.99 * torch.rand((num_parallel, 1, 1, design_dim))

            elapsed = time.time() - t
            print('elapsed design time', elapsed)
            results['design_time'] = elapsed
            results['d_star_design'] = d_star_design
            print('design', d_star_design.squeeze(), d_star_design.shape)
            d_star_designs = torch.cat([d_star_designs, d_star_design], dim=-2)
            y = true_model(d_star_design)
            ys = torch.cat([ys, y], dim=-1)
            print('ys', ys.squeeze(), ys.shape)
            results['y'] = y

            elbo_learn(
                prior, d_star_designs, ["y"], ["rho", "alpha", "slope"], elbo_n_samples, elbo_n_steps,
                partial(elboguide, dim=num_parallel), {"y": ys}, optim.Adam({"lr": elbo_lr})
            )
            rho_concentration = pyro.param("rho_concentration").detach().data.clone()
            alpha_concentration = pyro.param("alpha_concentration").detach().data.clone()
            slope_mu = pyro.param("slope_mu").detach().data.clone()
            slope_sigma = pyro.param("slope_sigma").detach().data.clone()
            print("rho_concentration", rho_concentration, "alpha_concentration", alpha_concentration, "slope_mu", slope_mu,
                  "slope_sigma", slope_sigma)
            results['rho_concentration'], results['alpha_concentration'] = rho_concentration, alpha_concentration
            results['slope_mu'], results['slope_sigma'] = slope_mu, slope_sigma

            with open(results_file, 'ab') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CES (Constant Elasticity of Substitution) indifference"
                                                 " iterated experiment design")
    parser.add_argument("--num-steps", nargs="?", default=25, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--typs", nargs="?", default="rand", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--lengthscale", nargs="?", default=10., type=float)
    args = parser.parse_args()
    main(args.num_steps, args.num_parallel, args.name, args.typs, args.seed, args.lengthscale)
