from __future__ import absolute_import, division, print_function

import torch
import pytest

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import Trace_ELBO
from pyro.contrib.oed.eig import (
    naive_rainforth_eig, barber_agakov_ape, gibbs_y_eig, gibbs_y_re_eig, iwae_eig, laplace_vi_ape, lfire_eig,
    donsker_varadhan_eig)
from pyro.contrib.util import rvv
from tests.common import assert_equal


@pytest.fixture
def finite_space_model():
    def model(design):
        theta = pyro.sample("theta", dist.Bernoulli(.4))
        y = pyro.sample("y", dist.Bernoulli((design + theta) / 2.))
        return y
    return model


@pytest.fixture
def one_point_design():
    return torch.tensor(.5)


@pytest.fixture
def true_ape():
    return torch.tensor(0.45628762737892914)


@pytest.fixture
def true_eig():
    return torch.tensor(0.21672403963032738)


def posterior_guide(y_dict, design, observation_labels, target_labels):

    y = torch.cat(list(y_dict.values()), dim=-1)
    a, b = pyro.param("a", torch.tensor(0.)), pyro.param("b", torch.tensor(0.))
    print(a,b)
    pyro.sample("theta", dist.Bernoulli(logits=a + b*y))


def marginal_guide(design, observation_labels, target_labels):

    logit_p = pyro.param("logit_p", torch.tensor(0.))
    print(logit_p)
    pyro.sample("y", dist.Bernoulli(logits=logit_p))


def likelihood_guide(theta_dict, design, observation_labels, target_labels):

    theta = torch.cat(list(theta_dict.values()), dim=-1)
    a, b = pyro.param("a", torch.tensor(0.)), pyro.param("b", torch.tensor(0.))
    pyro.sample("y", dist.Bernoulli(logits=a + b*theta))


def make_lfire_classifier(n_theta_samples):
    def lfire_classifier(design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)
        a, b = pyro.param("a", torch.zeros(n_theta_samples)), pyro.param("b", torch.zeros(n_theta_samples))

        return a + b*y

    return lfire_classifier


def dv_critic(design, trace, observation_labels, target_labels):
    y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
    theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
    x = torch.cat(list(theta_dict.values()) + list(y_dict.values()), dim=-1)

    a, b = pyro.param("a", torch.zeros(2)), pyro.param("b", torch.tensor(0.))

    return rvv(a, x) + b


########################################################################################################################
# TESTS
########################################################################################################################


def test_posterior_finite_space_model(finite_space_model, one_point_design, true_ape):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    barber_agakov_ape(finite_space_model, one_point_design, "y", "theta", num_samples=100,
                      num_steps=1000, guide=posterior_guide,
                      optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_ape = barber_agakov_ape(finite_space_model, one_point_design, "y", "theta", num_samples=100,
                                      num_steps=1000, guide=posterior_guide,
                                      optim=optim.Adam({"lr": 0.001}), final_num_samples=2000)
    assert_equal(estimated_ape, true_ape, prec=5e-2)


def test_marginal_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    gibbs_y_eig(finite_space_model, one_point_design, "y", "theta", num_samples=100,
                num_steps=500, guide=marginal_guide,
                optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = gibbs_y_eig(finite_space_model, one_point_design, "y", "theta", num_samples=100,
                                num_steps=50000, guide=marginal_guide,
                                optim=optim.Adam({"lr": 0.0001}), final_num_samples=500)
    assert_equal(estimated_eig, true_eig, prec=5e-2)
#
#
# def test_marginal_likelihood_finite_space_model(finite_space_model, one_point_design):
#     pyro.set_rng_seed(42)
#     pyro.clear_param_store()
#     # Pre-train (large learning rate)
#     gibbs_y_re_eig(finite_space_model, one_point_design, "y", "w", num_samples=10,
#                 num_steps=250, marginal_guide=marginal_guide, likelihood_guide=likelihood_guide,
#                 optim=optim.Adam({"lr": 0.1}))
#     # Finesse (small learning rate)
#     estimated_eig = gibbs_y_re_eig(finite_space_model, one_point_design, "y", "w", num_samples=10,
#                                 num_steps=250, marginal_guide=marginal_guide, likelihood_guide=likelihood_guide,
#                                 optim=optim.Adam({"lr": 0.01}), final_num_samples=500)
#     expected_eig = finite_space_model_ground_truth(finite_space_model, one_point_design, "y", "w")
#     assert_equal(estimated_eig, expected_eig, prec=5e-2)
#
#
# def test_vnmc_finite_space_model(finite_space_model, one_point_design):
#     pyro.set_rng_seed(42)
#     pyro.clear_param_store()
#     # Pre-train (large learning rate)
#     iwae_eig(finite_space_model, one_point_design, "y", "w", num_samples=[9, 3],
#              num_steps=250, guide=posterior_guide,
#              optim=optim.Adam({"lr": 0.1}))
#     # Finesse (small learning rate)
#     estimated_eig = iwae_eig(finite_space_model, one_point_design, "y", "w", num_samples=[9, 3],
#                              num_steps=250, guide=posterior_guide,
#                              optim=optim.Adam({"lr": 0.01}), final_num_samples=[500, 100])
#     expected_eig = finite_space_model_ground_truth(finite_space_model, one_point_design, "y", "w")
#     assert_equal(estimated_eig, expected_eig, prec=5e-2)
#
#
def test_naive_rainforth_eig_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = naive_rainforth_eig(finite_space_model, one_point_design, "y", "theta", M=100, N=100*100)
    assert_equal(estimated_eig, true_eig, prec=5e-2)


# def test_lfire_finite_space_model(finite_space_model, one_point_design):
#     pyro.set_rng_seed(42)
#     pyro.clear_param_store()
#     estimated_eig = lfire_eig(finite_space_model, one_point_design, "y", "w", num_y_samples=2, num_theta_samples=50,
#                               num_steps=1200, classifier=make_lfire_classifier(50), optim=optim.Adam({"lr": 0.0025}),
#                               final_num_samples=100)
#     expected_eig = finite_space_model_ground_truth(finite_space_model, one_point_design, "y", "w")
#     assert_equal(estimated_eig, expected_eig, prec=5e-2)
#
#
# def test_dv_finite_space_model(finite_space_model, one_point_design):
#     pyro.set_rng_seed(42)
#     pyro.clear_param_store()
#     donsker_varadhan_eig(finite_space_model, one_point_design, "y", "w", num_samples=100, num_steps=500, T=dv_critic,
#                          optim=optim.Adam({"lr": 0.1}))
#     estimated_eig = donsker_varadhan_eig(finite_space_model, one_point_design, "y", "w", num_samples=100,
#                                          num_steps=500, T=dv_critic, optim=optim.Adam({"lr": 0.001}),
#                                          final_num_samples=2000)
#     expected_eig = finite_space_model_ground_truth(finite_space_model, one_point_design, "y", "w")
#     assert_equal(estimated_eig, expected_eig, prec=5e-2)
