from __future__ import absolute_import, division, print_function

from collections import namedtuple
import time
import torch
import pytest
import numpy as np
from functools import partial

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import (
    vi_ape, naive_rainforth_eig, accelerated_rainforth_eig, donsker_varadhan_eig, barber_agakov_ape, gibbs_y_eig,
    gibbs_y_re_eig
)
from pyro.contrib.util import lexpand
from pyro.contrib.oed.util import (
    linear_model_ground_truth, vi_eig_lm, ba_eig_lm, ba_eig_mc, normal_inverse_gamma_ground_truth
)
from pyro.contrib.glmm import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix,
    normal_inverse_gamma_linear_model, normal_inverse_gamma_guide, group_linear_model,
    group_normal_guide, sigmoid_model_gamma, sigmoid_model_fixed, rf_group_assignments,
    known_covariance_linear_model, logistic_regression_model
)
from pyro.contrib.glmm.guides import (
    LinearModelGuide, NormalInverseGammaGuide, SigmoidGuide, GuideDV, LogisticGuide,
    LogisticResponseEst, LogisticCondResponseEst, SigmoidResponseEst, SigmoidCondResponseEst,
    NormalResponseEst, NormalCondResponseEst
)

PLOT = True

"""
Expected information gain estimation benchmarking
-------------------------------------------------
Models for benchmarking:

- A/B test: linear model with known variances and a discrete design on {0, ..., 10}
- linear model: classical linear model with designs on unit circle
- linear model with two parameter groups, aiming to learn just one
- A/B test with unknown observation covariance:
  - aim to learn regression coefficients *and* obs_sd
  - aim to learn regression coefficients, information on obs_sd ignored
- sigmoid model
- logistic regression*
- LMER with normal response and known obs_sd:*
  - aim to learn all unknowns: w, u and G_u*
  - aim to learn w*
  - aim to learn u*
- logistic-LMER*

* to do

Estimation techniques:

- analytic EIG, for linear models with known variances
- iterated variational inference with entropy
- naive Rainforth (nested Monte Carlo)
- Donsker-Varadhan
- Barber-Agakov

TODO:

- better guides- allow different levels of amortization
- SVI with BA-style guides
"""

#########################################################################################
# Designs
#########################################################################################
# All design tensors have shape: batch x n x p
# AB test
AB_test_11d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in range(0, 11)])
AB_test_11d_50n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 20-n])) for n in np.linspace(0, 20, 11)])
AB_test_2d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])
AB_test_11d_10n_12p = torch.cat([AB_test_11d_10n_2p, lexpand(torch.eye(10), 11)], dim=-1)

# Design on S^1
item_thetas = torch.linspace(0., np.pi, 10).unsqueeze(-1)
X_circle_10d_1n_2p = torch.stack([item_thetas.cos(), -item_thetas.sin()], dim=-1)
item_thetas_small = torch.linspace(0., np.pi/2, 5).unsqueeze(-1)
X_circle_5d_1n_2p = torch.stack([item_thetas_small.cos(), -item_thetas_small.sin()], dim=-1)

# Location finding designs
loc_15d_1n_2p = torch.stack([torch.linspace(-30., 30., 15), torch.ones(15)], dim=-1).unsqueeze(-2)
loc_4d_1n_2p = torch.tensor([[-5., 1], [-4.9, 1.], [4.9, 1], [5., 1.]]).unsqueeze(-2)

#########################################################################################
# Models
#########################################################################################
# Linear models
basic_2p_linear_model_sds_10_2pt5, basic_2p_guide = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]))
_, basic_2p_guide_w1 = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]), coef_label="w1")
basic_2p_linear_model_sds_10_0pt1, _ = zero_mean_unit_obs_sd_lm(torch.tensor([10., .1]))
basic_2p_ba_guide = lambda d: LinearModelGuide(d, {"w": 2})  # noqa: E731
normal_response_est = lambda d: NormalResponseEst(d, {"y": 10})
normal_response_est_50 = lambda d: NormalResponseEst(d, {"y": 20})
normal_cond_response_est = lambda d: NormalCondResponseEst(d, {"ab": 2, "re": 10}, {"y": 10})
normal_cond_response_est2 = lambda d: NormalCondResponseEst(d, {"w": 2}, {"y": 10})
group_2p_linear_model_sds_10_2pt5 = group_linear_model(torch.tensor(0.), torch.tensor([10.]), torch.tensor(0.),
                                                       torch.tensor([2.5]), torch.tensor(1.))
normal_re = group_linear_model(torch.tensor(0.), torch.tensor([10., .1]), torch.tensor(0.),
                               torch.ones(10), torch.tensor(1.), coef1_label="ab", coef2_label="re")
group_2p_guide = group_normal_guide(torch.tensor(1.), (1,), (1,))
group_2p_ba_guide = lambda d: LinearModelGuide(d, {"w1": 1, "w2": 1})  # noqa: E731
nig_2p_linear_model_3_2 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, 10.]),
                                                            torch.tensor([3.]), torch.tensor([2.]))
nig_2p_linear_model_15_14 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, 10.]),
                                                              torch.tensor([15.]), torch.tensor([14.]))
nig_re_3_2 = normal_inverse_gamma_linear_model([torch.tensor(0.), torch.tensor(0.)],
                                               [torch.tensor([.1, 10.]), torch.ones(10)],
                                               torch.tensor([3.]), torch.tensor([2.]),
                                               coef_labels=["ab", "re"])

re_guide = lambda d: LinearModelGuide(d, {"ab": 2, "re": 10})
nig_2p_guide = normal_inverse_gamma_guide((2,), mf=True)
nig_2p_ba_guide = lambda d: NormalInverseGammaGuide(d, {"w": 2})  # noqa: E731
nig_2p_ba_mf_guide = lambda d: NormalInverseGammaGuide(d, {"w": 2}, mf=True)  # noqa: E731

sigmoid_2p_model = sigmoid_model_fixed(torch.tensor([1., 10.]), torch.tensor([.25, 8.]),
                                       torch.tensor(2.))
sigmoid_re_model = sigmoid_model_fixed([torch.tensor([1.]), torch.tensor([10.])],
                                                  [torch.tensor([.25]), torch.tensor([8.])],
                                                  torch.tensor(2.), coef_labels=["coef", "loc"])
loc_2p_model = known_covariance_linear_model(torch.tensor([1., 10.]), torch.tensor([.25, 8.]), torch.tensor(2.),
                                             coef_label="w1")
logistic_2p_model = logistic_regression_model(torch.tensor([1., 10.]), torch.tensor([.25, 8.]), coef_labels="w1")
logistic_random_effects = logistic_regression_model([torch.tensor([1.]), torch.tensor([10.])],
                                                    [torch.tensor([.25]), torch.tensor([8.])],
                                                    coef_labels=["coef", "loc"])
loc_ba_guide = lambda d: LinearModelGuide(d, {"w1": 2})  # noqa: E731
logistic_guide  = lambda d: LogisticGuide(d, {"w1": 2})
logistic_random_effect_guide = lambda d: LogisticGuide(d, {"loc": 1})
logistic_response_est = lambda d: LogisticResponseEst(d, ["y"])
logistic_cond_response_est = lambda d: LogisticCondResponseEst(d, {"coef": 1, "loc": 1}, ["y"])
sigmoid_response_est = lambda d: SigmoidResponseEst(d, ["y"])
sigmoid_cond_response_est = lambda d: SigmoidCondResponseEst(d, {"coef": 1, "loc": 1}, ["y"])
sigmoid_low_guide = lambda d: SigmoidGuide(d, {"w": 2})  # noqa: E731
sigmoid_high_guide = lambda d: SigmoidGuide(d, {"w": 2})  # noqa: E731
sigmoid_random_effect_guide = lambda d: SigmoidGuide(d, {"coef": 1, "loc": 1})

########################################################################################
# Aux
########################################################################################

elbo = TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss

def zerofn(*args, **kwargs):
    return torch.tensor(0.)

zerofn.name = 'zero'

ba_lm_use_ae = partial(ba_eig_lm, analytic_entropy=True)
ba_lm_use_ae.__name__ = 'ba_lm_use_ae'

# Makes the plots look pretty
vi_eig_lm.name = "Variational inference"
vi_ape.name = "Variational inference"
ba_eig_lm.name = "Posterior"
ba_lm_use_ae.name = "Posterior, with analytic entropy"
ba_eig_mc.name = "Posterior"
barber_agakov_ape.name = "Posterior"
donsker_varadhan_eig.name = "Donsker-Varadhan"
linear_model_ground_truth.name = "Ground truth"
normal_inverse_gamma_ground_truth.name = "Ground truth"
naive_rainforth_eig.name = "Nested Monte Carlo"
accelerated_rainforth_eig.name = "Accelerated Rainforth"
gibbs_y_eig.name = "Marginal"
gibbs_y_re_eig.name = "Marginal with random effects"

T = namedtuple("CompareEstimatorsExample", [
    "title",
    "model",
    "design",
    "observation_label",
    "target_label",
    "arglist"
])

CMP_TEST_CASES = [
    T(
        "Sigmoid with random effects",
        sigmoid_re_model,
        loc_15d_1n_2p,
        "y",
        "loc",
        [  
            #(naive_rainforth_eig, [300*300, 300, 300, True]),
            (naive_rainforth_eig, [50*50, 50, 50, True]),
            (gibbs_y_re_eig,
             [10, 3000, sigmoid_response_est((10, 15)), sigmoid_cond_response_est((10, 15)),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (ba_eig_mc,
             [10, 1500, sigmoid_random_effect_guide((10, 15)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (naive_rainforth_eig, [150*150, 150, 150, True]),
        ]
    ),  
    T(
        "Sigmoid regression model",
        sigmoid_2p_model,
        loc_15d_1n_2p,
        "y",
        "w",
        [
            (naive_rainforth_eig, [70*70, 70]),
            (ba_eig_mc,
             [10, 400, sigmoid_high_guide((10, 15)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (gibbs_y_eig,
             [20, 4000, sigmoid_response_est((10, 15)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (naive_rainforth_eig, [300*300, 300]),
            #(donsker_varadhan_eig,
            # [400, 80, GuideDV(sigmoid_high_guide(15)),
            #  optim.Adam({"lr": 0.05}), False, None, 500])
        ]
    ),
    T(
        "Linear regression model",
        basic_2p_linear_model_sds_10_0pt1,
        AB_test_11d_10n_2p,
        "y",
        "w",
        [
            (ba_eig_lm,
             [10, 1200, basic_2p_ba_guide((10, 11)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (ba_lm_use_ae,
             [10, 1200, basic_2p_ba_guide((10, 11)), optim.Adam({"lr": 0.05}),
              False, None, 50]),
            (donsker_varadhan_eig,
             [100, 100, GuideDV(basic_2p_ba_guide((10, 11))),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (linear_model_ground_truth, []),
        ]
    ),
    T(
        "Normal inverse gamma model",
        nig_2p_linear_model_3_2,
        AB_test_11d_10n_2p,
        "y",
        ["w", "tau"],
        [
            (naive_rainforth_eig, [110*110, 110]),
            (ba_eig_mc,
             [10, 800, nig_2p_ba_mf_guide((10, 11)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (gibbs_y_eig,
             [10, 1200, normal_response_est((10, 11)),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (ba_eig_mc,
             [20, 1300, nig_2p_ba_guide((1, 11)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            # (normal_inverse_gamma_ground_truth, [])
            #(ba_eig_mc,
            # [20, 1600, nig_2p_ba_guide((10, 11)), optim.Adam({"lr": 0.05}),
            #  False, None, 500]),
        ]
    ),
    T(
        "Linear model with random effects",
        normal_re,
        lexpand(AB_test_11d_10n_12p, 10),
        "y",
        "ab",
        [
            (naive_rainforth_eig, [52*52, 52, 52, True]),
            (ba_eig_mc,
             [10, 150, re_guide((10, 11)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (gibbs_y_re_eig,
             [10, 600, normal_response_est((10, 11)), normal_cond_response_est((10, 11)),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (linear_model_ground_truth, []),
        ]
    ),
    T(
        "Linear regression model (large n)",
        basic_2p_linear_model_sds_10_0pt1,
        lexpand(AB_test_11d_50n_2p, 10),
        "y",
        "w",
        [
            (naive_rainforth_eig, [90*90, 90]),
            (ba_eig_lm,
             [10, 1000, basic_2p_ba_guide((10, 11)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (gibbs_y_eig,
             [10, 700, normal_response_est_50((10, 11)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            #(vi_eig_lm,
            # [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
            #   "num_steps": 1000}, {"num_samples": 1}]),
            #(donsker_varadhan_eig,
            # [400, 400, GuideDV(basic_2p_ba_guide((11,))),
            #  optim.Adam({"lr": 0.05}), False, None, 500]),
            (linear_model_ground_truth, []),
        ]
    ),
    T(
        "Sigmoid regression model",
        sigmoid_2p_model,
        lexpand(loc_15d_1n_2p, 10),
        "y",
        "w",
        [
            (naive_rainforth_eig, [75*75, 75]),
            (gibbs_y_eig,
             [20, 2800, sigmoid_response_est((10, 15)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (ba_eig_mc,
             [10, 180, sigmoid_high_guide((10, 15)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            #(donsker_varadhan_eig,
            # [400, 80, GuideDV(sigmoid_high_guide(15)),
            #  optim.Adam({"lr": 0.05}), False, None, 500])
        ]
    ),
    T(
        "Normal inverse gamma version of A/B test, with random effects",
        nig_re_3_2,
        lexpand(AB_test_11d_10n_12p, 10),
        "y",
        "ab",
        [
            (gibbs_y_re_eig,
             [10, 1100, normal_response_est((10, 11)), normal_cond_response_est((10, 11)),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (naive_rainforth_eig, [50*50, 50, 50, True]),
            (ba_eig_mc,
             [20, 600, re_guide((10, 11)), optim.Adam({"lr": 0.05}),
              False, None, 500])
        ]
    ),
    T(
        "Sigmoid with random effects",
        sigmoid_re_model,
        lexpand(loc_15d_1n_2p, 10),
        "y",
        "loc",
        [  
            (gibbs_y_re_eig,
             [40, 1600, sigmoid_response_est((10, 15)), sigmoid_cond_response_est((10, 15)),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (ba_eig_mc,
             [10, 400, sigmoid_random_effect_guide((10, 15)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (naive_rainforth_eig, [60*60, 60, 60, True])
        ]
    ),  
    T(
        "Logistic with random effects",
        logistic_random_effects,
        loc_15d_1n_2p,
        "y",
        "loc",
        [
            (accelerated_rainforth_eig, [{"y": torch.tensor([0., 1.])}, 100, 100]),
            (gibbs_y_re_eig,
             [40, 1200, logistic_response_est((15,)), logistic_cond_response_est((15,)),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (ba_eig_mc,
             [40, 800, logistic_random_effect_guide((15,)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
        ]
    ),
    T(
        "Logistic regression",
        logistic_2p_model,
        loc_15d_1n_2p,
        "y",
        "w1",
        [
            (accelerated_rainforth_eig, [{"y": torch.tensor([0., 1.])}, 2000]),
            (gibbs_y_eig,
             [40, 400, logistic_response_est((15,)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (ba_eig_mc,
             [40, 800, logistic_guide((15,)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            # (donsker_varadhan_eig,
            # [400, 400, GuideDV(logistic_guide(15)),
            #  optim.Adam({"lr": 0.05}), False, None, 500]),
        ]
    ),
    T(
        "A/B test linear model with known observation variance",
        basic_2p_linear_model_sds_10_2pt5,
        AB_test_11d_10n_2p,
        "y",
        "w",
        [
            (linear_model_ground_truth, []),
            (naive_rainforth_eig, [2000, 2000]),
            (vi_eig_lm,
             [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
               "num_steps": 1000}, {"num_samples": 1}]),
            (donsker_varadhan_eig,
             [400, 800, GuideDV(basic_2p_ba_guide((11,))),
              optim.Adam({"lr": 0.025}), False, None, 500]),
            (ba_eig_lm,
             [20, 400, basic_2p_ba_guide((11,)), optim.Adam({"lr": 0.05}),
              False, None, 500])
        ]
    ),
    # TODO: make this example work better
    T(
        "A/B testing with unknown covariance (Gamma(15, 14))",
        nig_2p_linear_model_15_14,
        AB_test_11d_10n_2p,
        "y",
        ["w", "tau"],
        [
            (naive_rainforth_eig, [2000, 2000]),
            (vi_ape,
             [{"guide": nig_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
               "num_steps": 1000}, {"num_samples": 4}]),
            (barber_agakov_ape,
             [20, 800, nig_2p_ba_guide((11,)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (barber_agakov_ape,
             [20, 800, nig_2p_ba_mf_guide((11,)), optim.Adam({"lr": 0.05}),
              False, None, 500])
        ]
    ),
    # TODO: make this example work better
    T(
        "A/B testing with unknown covariance (Gamma(3, 2))",
        nig_2p_linear_model_3_2,
        AB_test_11d_10n_2p,
        "y",
        ["w", "tau"],
        [
            (naive_rainforth_eig, [2000, 2000]),
            (vi_ape,
             [{"guide": nig_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
               "num_steps": 1000}, {"num_samples": 4}]),
            (barber_agakov_ape,
             [20, 800, nig_2p_ba_guide((11,)), optim.Adam({"lr": 0.05}),
              False, None, 500]),
            (barber_agakov_ape,
             [20, 800, nig_2p_ba_mf_guide((11,)), optim.Adam({"lr": 0.05}),
              False, None, 500])
        ]
    ),
    # TODO: make VI work here (non-mean-field guide)
    T(
        "Linear model targeting one parameter",
        group_2p_linear_model_sds_10_2pt5,
        X_circle_10d_1n_2p,
        "y",
        "w1",
        [
            (linear_model_ground_truth, []),
            (naive_rainforth_eig, [200, 200, 200]),
            (vi_eig_lm,
             [{"guide": group_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
               "num_steps": 1000}, {"num_samples": 1}]),
            (donsker_varadhan_eig,
             [400, 400, GuideDV(group_2p_ba_guide((10,))),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (ba_eig_lm,
             [20, 400, group_2p_ba_guide((10,)), optim.Adam({"lr": 0.05}),
              False, None, 500])
        ]
    ),
    T(
        "Linear model with designs on S^1",
        basic_2p_linear_model_sds_10_2pt5,
        X_circle_10d_1n_2p,
        "y",
        "w",
        [
            (linear_model_ground_truth, []),
            (naive_rainforth_eig, [2000, 2000]),
            (vi_eig_lm,
             [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
               "num_steps": 1000}, {"num_samples": 1}]),
            (donsker_varadhan_eig,
             [400, 400, GuideDV(basic_2p_ba_guide((10,))),
              optim.Adam({"lr": 0.05}), False, None, 500]),
            (ba_eig_lm,
             [20, 400, basic_2p_ba_guide((10,)), optim.Adam({"lr": 0.05}),
              False, None, 500])
        ]
    ),
]


@pytest.mark.parametrize("title,model,design,observation_label,target_label,arglist", CMP_TEST_CASES)
def test_eig_and_plot(title, model, design, observation_label, target_label, arglist):
    """
    Runs a group of EIG estimation tests and plots the estimates on a single set
    of axes. Typically, each test within one `arglist` should estimate the same quantity.
    This is repeated for each `arglist`.
    """
    ep = 0.05
    ys = []
    means = []
    sds = []
    names = []
    elapseds = []
    markers = ['x', '+', 'o', 'D', 'v', '^']
    print(title)
    for n, (estimator, args) in enumerate(arglist):
        y, elapsed = time_eig(estimator, model, lexpand(design, 10) if n<(len(arglist)-1) else lexpand(design,1), observation_label, target_label, args)
        y = y.detach().numpy()
        y[np.isinf(y)] = np.nan
        ys.append(y)
        means.append(np.nanmean(y, 0))
        sds.append(2*np.nanstd(y, 0))
        elapseds.append(elapsed)
        names.append(estimator.name)

    bias = [np.nanmean(m - means[-1]) for m in means]
    variance = [np.nanmean(s - sds[-1]) for s in sds]
    print(bias, variance)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        x = np.arange(0, means[0].shape[0])
        for n, (y, s) in enumerate(zip(means[:-1], sds[:-1])):
            plt.errorbar(x+n*ep, y-means[-1], yerr=s, linestyle='-', marker=markers[n], markersize=10)
        plt.title(title, fontsize=18)
        plt.legend(names, loc=2, fontsize=16)
        plt.axhline(color='k')
        plt.xlabel("Design", fontsize=18)
        plt.ylabel("EIG estimation error", fontsize=18)
        plt.show()

        # newx = np.arange(0, len(elapseds))
        # plt.bar(newx, np.array(elapseds), color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
        # plt.gca().set_xticks(newx)
        # plt.gca().set_xticklabels(names)
        # plt.show()


def time_eig(estimator, model, design, observation_label, target_label, args):
    pyro.clear_param_store()

    t = time.time()
    y = estimator(model, design, observation_label, target_label, *args)
    elapsed = time.time() - t

    print(estimator.__name__)
    print('estimate', y)
    print('elapsed', elapsed)
    return y, elapsed


U = namedtuple("CheckConvergenceExample", [
    "title",
    "model",
    "design",
    "observation_label",
    "target_label",
    "est1",
    "est2",
    "kwargs1",
    "kwargs2"
])

CONV_TEST_CASES = [
    U(
        "Logistic regression",
        logistic_2p_model,
        loc_4d_1n_2p,
        "y", "w1",
        barber_agakov_ape,
        None,
        {"num_steps": 800, "num_samples": 40, "optim": optim.Adam({"lr": 0.05}),
         "guide": logistic_guide((4,)), "final_num_samples": 1000},
        {}
    ),
    # U(
    #     "Location finding -- no sigmoid -- sanity check",
    #     loc_2p_model,
    #     loc_4d_1n_2p,
    #     "y", "w1",
    #     barber_agakov_ape,
    #     linear_model_ground_truth,
    #     {"num_steps": 800, "num_samples": 40, "optim": optim.Adam({"lr": 0.05}),
    #      "guide": loc_ba_guide(4), "final_num_samples": 1000},
    #     {"eig": False}
    # ),
    # U(
    #     "Low slope sigmoid -- should reproduce linear model",
    #     sigmoid_low_2p_model,
    #     loc_4d_1n_2p,
    #     "y", "w1",
    #     barber_agakov_ape,
    #     linear_model_ground_truth,
    #     {"num_steps": 800, "num_samples": 40, "optim": optim.Adam({"lr": 0.05}),
    #      "guide": sigmoid_low_guide(4), "final_num_samples": 1000},
    #     {"eig": False}
    # ),
    U(
        "High slope sigmoid -- should see major difference",
        sigmoid_2p_model,
        loc_4d_1n_2p,
        "y", "w1",
        barber_agakov_ape,
        linear_model_ground_truth,
        {"num_steps": 800, "num_samples": 40, "optim": optim.Adam({"lr": 0.05}),
         "guide": sigmoid_high_guide((4,)), "final_num_samples": 1000},
        {"eig": False}
    ),
    U(
        "Barber-Agakov on A/B test with unknown covariance",
        nig_2p_linear_model_3_2,
        AB_test_2d_10n_2p,
        "y",
        ["w", "tau"],
        barber_agakov_ape,
        None,
        {"num_steps": 800, "num_samples": 20, "optim": optim.Adam({"lr": 0.05}),
         "guide": nig_2p_ba_guide((2,)), "final_num_samples": 1000},
        {}
    ),
    U(
        "Barber-Agakov on A/B test with unknown covariance (mean-field guide)",
        nig_2p_linear_model_3_2,
        AB_test_2d_10n_2p,
        "y",
        ["w", "tau"],
        barber_agakov_ape,
        None,
        {"num_steps": 800, "num_samples": 20, "optim": optim.Adam({"lr": 0.05}),
         "guide": nig_2p_ba_mf_guide((2,)), "final_num_samples": 1000},
        {}
    ),
    U(
        "Barber-Agakov on circle",
        basic_2p_linear_model_sds_10_2pt5,
        X_circle_5d_1n_2p,
        "y",
        "w",
        barber_agakov_ape,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 10, "optim": optim.Adam({"lr": 0.05}),
         "guide": basic_2p_ba_guide((5,)), "final_num_samples": 1000},
        {"eig": False}
    ),
    U(
        "Barber-Agakov on small AB test",
        basic_2p_linear_model_sds_10_2pt5,
        AB_test_2d_10n_2p,
        "y",
        "w",
        barber_agakov_ape,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 10, "optim": optim.Adam({"lr": 0.05}),
         "guide": basic_2p_ba_guide((2,)), "final_num_samples": 1000},
        {"eig": False}
    ),
    U(
        "Donsker-Varadhan on small AB test",
        basic_2p_linear_model_sds_10_2pt5,
        AB_test_2d_10n_2p,
        "y",
        "w",
        donsker_varadhan_eig,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 100, "optim": optim.Adam({"lr": 0.05}),
         "T": GuideDV(basic_2p_ba_guide((2,))), "final_num_samples": 10000},
        {}
    ),
    U(
        "Donsker-Varadhan on circle",
        basic_2p_linear_model_sds_10_2pt5,
        X_circle_5d_1n_2p,
        "y",
        "w",
        donsker_varadhan_eig,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 400, "optim": optim.Adam({"lr": 0.05}),
         "T": GuideDV(basic_2p_ba_guide((5,))), "final_num_samples": 10000},
        {}
    ),
]


@pytest.mark.parametrize("title,model,design,observation_label,target_label,est1,est2,kwargs1,kwargs2", CONV_TEST_CASES)
def test_convergence(title, model, design, observation_label, target_label, est1, est2, kwargs1, kwargs2):
    """
    Produces a convergence plot for a Barber-Agakov or Donsker-Varadhan
    EIG estimation.
    """
    t = time.time()
    pyro.clear_param_store()
    if est2 is not None:
        truth = est2(model, design, observation_label, target_label, **kwargs2)
    else:
        truth = None
    dv, final = est1(model, design, observation_label, target_label, return_history=True, **kwargs1)
    x = np.arange(0, dv.shape[0])
    print(est1.__name__)
    if truth is not None:
        print("Final est", final, "Truth", truth, "Error", (final - truth).abs().sum())
    else:
        print("Final est", final)
    print("Time", time.time() - t)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(x, dv.detach().numpy())

        if truth is not None:
            for true, col in zip(torch.unbind(truth, 0), plt.rcParams['axes.prop_cycle'].by_key()['color']):
                plt.axhline(true.numpy(), color=col)

        plt.title(title)
        plt.show()
