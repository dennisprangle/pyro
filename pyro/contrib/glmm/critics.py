from __future__ import absolute_import, division, print_function

import torch
from torch import nn

import pyro
from pyro.contrib.util import get_indices, rmv, rvv, rtril, lexpand, rexpand
from pyro.contrib.glmm.guides import NormalMarginalGuide, NormalLikelihoodGuide


class LinearModelAmortizedClassifier(nn.Module):

    def __init__(self, d, w_sizes, y_sizes, regressor_init=-3., i_scale_tril_init=3., **kwargs):
        super(LinearModelAmortizedClassifier, self).__init__()
        n = sum(y_sizes.values())
        p = sum(w_sizes.values())
        self.w_sizes = w_sizes
        self.bias = nn.Parameter(torch.zeros(*d))
        self.regressor = nn.Parameter(regressor_init * torch.ones(*d, p, n))
        self.i_scale_tril = nn.Parameter(i_scale_tril_init * lexpand(torch.eye(p), *d))
        self.softplus = nn.Softplus()

    def forward(self, design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
        theta = torch.cat(list(theta_dict.values()), dim=-1)

        posterior_mean = rmv(self.softplus(self.regressor), y)
        a = rmv(rtril(self.i_scale_tril, upper=True), theta - posterior_mean)
        return -rvv(a, a)


class LinearModelBootstrapClassifier(nn.Module):
    """This classifier uses the same parameters as the marginal + likelihood method. Specifically,
    it returns the likelihood ratio by calling q(y|d) and q(y|theta,d). The difference between using
    this and using marginal + likelihood is then just the choice of loss function.
    """

    def __init__(self, d, w_sizes, y_sizes, scale_tril_init=3., **kwargs):
        super(LinearModelBootstrapClassifier, self).__init__()
        self.marginal_guide = NormalMarginalGuide(d, y_sizes, sigma_init=scale_tril_init)
        self.likelihood_guide = NormalLikelihoodGuide(d, w_sizes, y_sizes, sigma_init=scale_tril_init)

    def forward(self, design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        # Run through q(y | d)
        qyd = pyro.condition(self.marginal_guide, data=y_dict)
        marginal_trace = pyro.poutine.trace(qyd).get_trace(
            design, observation_labels, target_labels)
        marginal_trace.compute_log_prob()

        # Run through q(y | theta, d)
        qythetad = pyro.condition(self.likelihood_guide, data=y_dict)
        cond_trace = pyro.poutine.trace(qythetad).get_trace(
            theta_dict, design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        terms = -sum(marginal_trace.nodes[l]["log_prob"] for l in observation_labels)
        terms += sum(cond_trace.nodes[l]["log_prob"] for l in observation_labels)
        return terms


class LinearModelClassifier(nn.Module):

    def __init__(self, d, ntheta, w_sizes, y_sizes, scale_tril_init=3., **kwargs):
        super(LinearModelClassifier, self).__init__()
        n = sum(y_sizes.values())
        self.w_sizes = w_sizes
        self.bias = nn.Parameter(torch.zeros(ntheta, *d))
        self.offset = nn.Parameter(torch.zeros(ntheta, *d, n))
        self.bilinear = nn.Parameter(scale_tril_init * lexpand(torch.eye(n), ntheta, *d))

    def forward(self, design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)

        a = rmv(rtril(self.bilinear, upper=True), y - self.offset)
        return self.bias - rvv(a, a)


class SigmoidLocationAmortizedClassifier(nn.Module):

    def __init__(self, d, w_sizes, y_sizes, multiplier, scale_tril_init=3., **kwargs):
        super(SigmoidLocationAmortizedClassifier, self).__init__()
        n = sum(y_sizes.values())
        self.w_sizes = w_sizes
        self.bias0 = nn.Parameter(torch.zeros(*d))
        self.bias1 = nn.Parameter(torch.zeros(*d))
        self.offset = nn.Parameter(torch.zeros(*d, n))
        self.offset0 = nn.Parameter(torch.zeros(*d, n))
        self.offset1 = nn.Parameter(torch.zeros(*d, n))
        self.bilinear = nn.Parameter(scale_tril_init * lexpand(torch.eye(n), *d))
        self.bilinear0 = nn.Parameter(scale_tril_init * lexpand(torch.eye(n), *d))
        self.bilinear1 = nn.Parameter(scale_tril_init * lexpand(torch.eye(n), *d))
        self.multiplier = multiplier

        # TODO read from torch float specs
        self.epsilon = torch.tensor(2 ** -24)

    def forward(self, design, trace, observation_labels, target_labels):
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
        theta = torch.cat(list(theta_dict.values()), dim=-1)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        test_point = rvv(design, self.multiplier)
        y = torch.cat(list(y_dict.values()), dim=-1)
        mask0 = (y <= self.epsilon).squeeze(-1).float()
        mask1 = (1. - y <= self.epsilon).squeeze(-1).float()
        y_trans = y.log() - (1. - y).log()
        eta = test_point - y_trans

        a = rmv(rtril(self.bilinear + rexpand(mask0, 1, 1) * self.bilinear0 + rexpand(mask1, 1, 1) * self.bilinear1,
                      upper=True),
                theta - eta + self.offset + rexpand(mask0, 1) * self.offset0 + rexpand(mask1, 1) * self.offset1)
        return -rvv(a, a) + mask0 * self.bias0 + mask1 * self.bias1


class SigmoidLocationClassifier(nn.Module):

    def __init__(self, d, ntheta, w_sizes, y_sizes, multiplier, scale_tril_init=3., **kwargs):
        super(SigmoidLocationClassifier, self).__init__()
        n = sum(y_sizes.values())
        self.w_sizes = w_sizes
        self.bias = nn.Parameter(torch.zeros(ntheta, *d))
        self.bias0 = nn.Parameter(torch.zeros(ntheta, *d))
        self.bias1 = nn.Parameter(torch.zeros(ntheta, *d))
        self.offset = nn.Parameter(torch.zeros(ntheta, *d, n))
        self.bilinear = nn.Parameter(scale_tril_init * lexpand(torch.eye(n), ntheta, *d))
        self.multiplier = multiplier

        # TODO read from torch float specs
        self.epsilon = torch.tensor(2 ** -24)

    def forward(self, design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        test_point = rvv(design, self.multiplier)
        y = torch.cat(list(y_dict.values()), dim=-1)
        mask0 = (y <= self.epsilon).squeeze(-1).float()
        mask1 = (1. - y <= self.epsilon).squeeze(-1).float()
        y_trans = y.log() - (1. - y).log()
        eta = test_point - y_trans

        a = rmv(rtril(self.bilinear, upper=True), eta - self.offset)
        return self.bias - rvv(a, a) + mask0 * self.bias0 + mask1 * self.bias1


class LogisticExtrapolationClassifier(nn.Module):

    def __init__(self, d, **kwargs):
        super(LogisticExtrapolationClassifier, self).__init__()
        self.weight_y = nn.Parameter(torch.zeros(*d))
        self.weight_t = nn.Parameter(torch.zeros(*d))
        self.weight_ty = nn.Parameter(torch.zeros(*d))

    def forward(self, design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1).squeeze(-1)
        t_dict = {l: trace.nodes[l]["value"] for l in target_labels}
        t = torch.cat(list(t_dict.values()), dim=-1).squeeze(-1)

        return y * self.weight_y + t * self.weight_t + y*t*self.weight_ty


class TurkAmortizedClassifier(nn.Module):

    def __init__(self, d, w_sizes, y_sizes, bilinear_init=0., **kwargs):
        super(TurkAmortizedClassifier, self).__init__()
        self.w_sizes = w_sizes
        n = sum(y_sizes.values())
        self.bilinear = nn.Parameter(bilinear_init * torch.ones(*d, n, n))
        self.log_multiplier = nn.Parameter(torch.zeros(*d, 1))

        # TODO read from torch float specs
        self.epsilon = torch.tensor(2 ** -24)

    def forward(self, design, trace, observation_labels, target_labels):
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)
        y_trans = y.log() - (1. - y).log()

        theta = torch.cat(list(theta_dict.values()), dim=-1)
        indices = get_indices(target_labels, self.w_sizes)
        subdesign = design[..., indices]
        centre = rmv(subdesign, theta)
        scaled_centre = torch.exp(self.log_multiplier) * centre

        a = rmv(rtril(self.bilinear, upper=True), y_trans - scaled_centre)
        return -rvv(a, a)


class TurkClassifier(nn.Module):

    def __init__(self, d, ntheta, w_sizes, y_sizes, bilinear_init=0., **kwargs):
        super(TurkClassifier, self).__init__()
        self.w_sizes = w_sizes
        n = sum(y_sizes.values())
        self.bilinear = nn.Parameter(bilinear_init * torch.ones(ntheta, *d))
        self.centre = nn.Parameter(torch.zeros(ntheta, *d, n))
        self.bias = nn.Parameter(torch.zeros(ntheta, *d))

        # TODO read from torch float specs
        self.epsilon = torch.tensor(2 ** -24)

    def forward(self, design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)
        y_trans = y.log() - (1. - y).log()

        a = y_trans - self.centre
        return self.bilinear * rvv(a, a) + self.bias
