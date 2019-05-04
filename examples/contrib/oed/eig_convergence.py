from __future__ import absolute_import, division, print_function

import argparse
import torch
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.contrib.oed.eig import donsker_varadhan_eig, gibbs_y_eig
from pyro.contrib.util import rmv, rvv
from pyro.contrib.oed.util import linear_model_ground_truth, ba_eig_lm
from pyro.contrib.glmm import known_covariance_linear_model


def posterior_guide(y_dict, design, observation_labels, target_labels):

    y = torch.cat(list(y_dict.values()), dim=-1)
    A = pyro.param("A", torch.zeros(2, 3))
    scale_tril = pyro.param("scale_tril", torch.tensor([[1., 0.], [0., 1.5]]),
                            constraint=torch.distributions.constraints.lower_cholesky)
    mu = rmv(A, y)
    pyro.sample("w", dist.MultivariateNormal(mu, scale_tril=scale_tril))


def marginal_guide(design, observation_labels, target_labels):

    mu = pyro.param("mu", torch.zeros(3))
    scale_tril = pyro.param("scale_tril", torch.eye(3),
                            constraint=torch.distributions.constraints.lower_cholesky)
    pyro.sample("y", dist.MultivariateNormal(mu, scale_tril))


def dv_critic(design, trace, observation_labels, target_labels):
    y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
    theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
    x = torch.cat(list(theta_dict.values()) + list(y_dict.values()), dim=-1)

    B = pyro.param("B", torch.zeros(5, 5))
    return rvv(x, rmv(B, x))


def main(num_steps, seed, plot):

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    linear_model = known_covariance_linear_model(coef_means=torch.tensor(0.),
                                                 coef_sds=torch.tensor([1., 1.5]),
                                                 observation_sd=torch.tensor(1.))
    # Basic design matrix
    X = torch.zeros(3, 2)
    X[0, 0] = X[1, 1] = X[2, 1] = 1.

    expected_eig = linear_model_ground_truth(linear_model, X, "y", "w")
    ba_eig, _ = ba_eig_lm(linear_model, X, "y", "w", num_samples=10,
                          num_steps=num_steps, guide=posterior_guide,
                          optim=optim.Adam({"lr": 0.01}), final_num_samples=500,
                          return_history=True)
    pyro.clear_param_store()
    marginal_eig, _ = gibbs_y_eig(linear_model, X, "y", "w", num_samples=10,
                                  num_steps=num_steps, guide=marginal_guide,
                                  optim=optim.Adam({"lr": 0.01}), final_num_samples=500,
                                  return_history=True)
    pyro.clear_param_store()
    dv_eig, _ = donsker_varadhan_eig(linear_model, X, "y", "w", num_samples=40,
                                     num_steps=num_steps, T=dv_critic, optim=optim.Adam({"lr": 0.005}),
                                     final_num_samples=500, return_history=True)

    if plot:
        import matplotlib.pyplot as plt
        x = np.arange(0, num_steps)
        plt.figure(figsize=(8, 5))
        plt.plot(x, ba_eig.detach().numpy())
        plt.plot(x, marginal_eig.detach().numpy())
        plt.plot(x, dv_eig.detach().numpy())

        plt.axhline(expected_eig.numpy(), color='k')
        plt.legend(["Posterior", "Marginal", "DV", "Ground truth"])
        plt.show()

    else:
        print(ba_eig)
        print(marginal_eig)
        print(dv_eig)
        print(expected_eig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIG convergence example")
    parser.add_argument("--num-steps", nargs="?", default=100, type=int)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.set_defaults(plot=False)
    args = parser.parse_args()
    main(args.num_steps, args.seed, args.plot)
