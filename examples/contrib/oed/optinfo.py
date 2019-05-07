from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.contrib.oed.eig import barber_agakov_ape
from pyro.contrib.util import rmv, iter_iaranges_to_shape, lexpand
from pyro.contrib.glmm import group_assignment_matrix

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


N = 15
prior_scale_tril = torch.tensor([[10., 0.], [0., .1]])
AB_test_designs = torch.stack([group_assignment_matrix(torch.tensor([n, N-n])) for n in torch.linspace(0, N, N+1)])


def model_learn_xi(design_prototype):
    thetas = pyro.param('xi', torch.zeros(design_prototype.shape[-2]))
    xi = lexpand(torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1), 1)
    batch_shape = design_prototype.shape[:-2]
    with ExitStack() as stack:
        for iarange in iter_iaranges_to_shape(batch_shape):
            stack.enter_context(iarange)

        x = pyro.sample("x", dist.MultivariateNormal(lexpand(torch.zeros(2), *batch_shape),
                                                     scale_tril=lexpand(prior_scale_tril, *batch_shape)))
        prediction_mean = rmv(xi, x)
        return pyro.sample("y", dist.Normal(prediction_mean, torch.tensor(1.)).independent(1))


def model_fix_xi(design):
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for iarange in iter_iaranges_to_shape(batch_shape):
            stack.enter_context(iarange)

        x = pyro.sample("x", dist.MultivariateNormal(lexpand(torch.zeros(2), *batch_shape),
                                                     scale_tril=lexpand(prior_scale_tril, *batch_shape)))
        prediction_mean = rmv(design, x)
        return pyro.sample("y", dist.Normal(prediction_mean, torch.tensor(1.)).independent(1))


def make_posterior_guide(d):
    def posterior_guide(y_dict, design, observation_labels, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        A = pyro.param("A", torch.zeros(d, 2, N))
        scale_tril = pyro.param("scale_tril", lexpand(prior_scale_tril, d),
                                constraint=torch.distributions.constraints.lower_cholesky)
        mu = rmv(A, y)
        pyro.sample("x", dist.MultivariateNormal(mu, scale_tril=scale_tril))
    return posterior_guide


def main():
    # Here we estimate the APE (average posterior entropy) on a grid of possible xi's
    # We are computing the `posterior' lower bound
    pyro.clear_param_store()
    ape_surf = barber_agakov_ape(model_fix_xi, AB_test_designs, "y", "x", guide=make_posterior_guide(N + 1),
                                 num_steps=6000, num_samples=10, optim=optim.Adam({"lr": 0.0025}),
                                 final_num_samples=500)
    pyro.clear_param_store()
    # Here we optimize xi and phi simultaneously
    ape_star = barber_agakov_ape(model_learn_xi, torch.zeros(1, N, 2), "y", "x", guide=make_posterior_guide(1),
                                 num_steps=60000, num_samples=10, optim=optim.Adam({"lr": 0.001}),
                                 final_num_samples=500)
    thetas = pyro.param("xi")
    xi = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1)
    print("Grid of designs", AB_test_designs.sum(-2))
    print("APE on grid (not learning xi)", ape_surf)
    print("Learned xi_star", xi.abs().sum(0))
    print("APE from xi_star", ape_star)


if __name__ == '__main__':
    main()
