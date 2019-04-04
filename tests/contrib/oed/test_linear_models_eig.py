from __future__ import absolute_import, division, print_function

import torch
import pytest

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.contrib.glmm import known_covariance_linear_model
from pyro.contrib.oed.util import linear_model_ground_truth
from pyro.contrib.oed.eig import naive_rainforth_eig, barber_agakov_ape
from pyro.contrib.util import rmv
from tests.common import assert_equal


@pytest.fixture
def linear_model():
    return known_covariance_linear_model(coef_means=torch.tensor(0.),
                                         coef_sds=torch.tensor([1., 1.5]),
                                         observation_sd=torch.tensor(1.))


@pytest.fixture
def one_point_design():
    X = torch.zeros(3, 2)
    X[0,0] = X[1,1] = X[2,1] = 1.
    return X


# Eight methods to test:
# posterior, marginal, vnmc, marginal+likelihood
# nmc, laplace, lfire, dv


def posterior_guide(y_dict, design, observation_labels, target_labels):

    y = torch.cat(list(y_dict.values()), dim=-1)
    A = pyro.param("A", torch.zeros(2, 3))
    scale_tril = pyro.param("scale_tril", torch.tensor([[1., 0.], [0., 1.5]]),
                            constraint=torch.distributions.constraints.lower_cholesky)
    mu = rmv(A, y)
    pyro.sample("w", dist.MultivariateNormal(mu, scale_tril=scale_tril))


def test_posterior_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    barber_agakov_ape(linear_model, one_point_design, "y", "w", num_samples=10,
                      num_steps=250, guide=posterior_guide,
                      optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_ape = barber_agakov_ape(linear_model, one_point_design, "y", "w", num_samples=10,
                                      num_steps=250, guide=posterior_guide,
                                      optim=optim.Adam({"lr": 0.01}), final_num_samples=500)
    expected_ape = linear_model_ground_truth(linear_model, one_point_design, "y", "w", eig=False)
    assert_equal(estimated_ape, expected_ape, prec=5e-2)


def test_naive_rainforth_eig_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = naive_rainforth_eig(linear_model, one_point_design, "y", "w", M=60, N=60*60)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)