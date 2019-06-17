from __future__ import absolute_import, division, print_function

import torch

from pyro.contrib.util import lexpand
from pyro.contrib.glmm import group_assignment_matrix, analytic_posterior_cov

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


N = 15
prior_scale_tril = torch.tensor([[10., 0.], [0., 2.]])


def linear_model(thetas):
    design = lexpand(torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1), 1)
    # print(design.abs().sum(1))
    x_dist = torch.distributions.MultivariateNormal(torch.zeros(2), scale_tril=prior_scale_tril)
    x = x_dist.sample()
    x_prime = x_dist.sample()
    eps_dist = torch.distributions.Normal(torch.zeros(N), torch.ones(N))
    eps = eps_dist.sample()

    prediction_mean = torch.matmul(design, x)
    y_dist = torch.distributions.MultivariateNormal(prediction_mean, scale_tril=torch.eye(N))
    y = prediction_mean + eps
    y_prime = torch.matmul(design, x_prime) + eps
    log_prob = y_dist.log_prob(y)

    g1 = torch.autograd.grad([log_prob], [y])
    # print(eps, g1)

    g2 = torch.autograd.grad([y], [thetas], grad_outputs=g1, retain_graph=True)[0]
    g3 = torch.autograd.grad([y_prime], [thetas], grad_outputs=g1)[0]

    return g3


def opt_mi():
    thetas = torch.zeros(N, requires_grad=True)
    optimizer = torch.optim.SGD([thetas], lr=0.005)
    for i in range(5000):
        thetas.grad = (1/10)*sum(linear_model(thetas) for _ in range(10))
        optimizer.step()
        design = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1)
        f = torch.logdet(analytic_posterior_cov(prior_scale_tril ** 2, design, 1.0))

    print(f)
    print(design.abs().sum(0))


def opt_true():
    thetas = torch.rand(N, requires_grad=True)
    optimizer = torch.optim.SGD([thetas], lr=0.005)
    for i in range(5000):
        optimizer.zero_grad()
        design = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1)
        #design.register_hook(print)

        f = torch.logdet(analytic_posterior_cov(prior_scale_tril ** 2, design, 1.0))
        f.backward()
        optimizer.step()

    print(f)
    print(design.abs().sum(0))


if __name__ == '__main__':

    opt_true()
    opt_mi()
