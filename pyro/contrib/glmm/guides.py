from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.distributions.transforms import AffineTransform, SigmoidTransform

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.util import get_indices, tensor_to_dict, rmv, rvv, rtril, rdiag
from pyro.ops.linalg import rinverse


class LinearModelGuide(nn.Module):

    def __init__(self, d, w_sizes, tikhonov_init=-2., scale_tril_init=3.):
        """
        Guide for linear models. No amortisation happens over designs.
        Amortisation over data is taken care of by analytic formulae for
        linear models (heavy use of truth).

        :param int d: the number of designs
        :param dict w_sizes: map from variable string names to int.
        :param float tikhonov_init: initial value for `tikhonov_diag` parameter.
        :param float scale_tril_init: initial value for `scale_tril` parameter.
        """
        super(LinearModelGuide, self).__init__()
        # Represent each parameter group as independent Gaussian
        # Making a weak mean-field assumption
        # To avoid this- combine labels
        self.tikhonov_diag = nn.Parameter(
                tikhonov_init*torch.ones(*d, sum(w_sizes.values())))
        self.scaled_prior_mean = nn.Parameter(torch.zeros(*d, sum(w_sizes.values())))
        self.scale_tril = {l: nn.Parameter(
                scale_tril_init*torch.ones(*d, p, p)) for l, p in w_sizes.items()}
        # This registers the dict values in pytorch
        # Await new version to use nn.ParamterDict
        self._registered = nn.ParameterList(self.scale_tril.values())
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        return self.linear_model_formula(y, design, target_labels)

    def linear_model_formula(self, y, design, target_labels):

        tikhonov_diag = rdiag(self.softplus(self.tikhonov_diag))
        xtx = torch.matmul(design.transpose(-1, -2), design) + tikhonov_diag
        xtxi = rinverse(xtx, sym=True)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y) + self.scaled_prior_mean)

        # Extract sub-indices
        mu = tensor_to_dict(self.w_sizes, mu, subset=target_labels)
        scale_tril = {l: rtril(self.scale_tril[l]) for l in target_labels}

        return mu, scale_tril

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("ba_guide", self)

        # Returns two dicts from labels -> tensors
        mu, scale_tril = self.get_params(y_dict, design, target_labels)

        for l in target_labels:
            w_dist = dist.MultivariateNormal(mu[l], scale_tril=scale_tril[l])
            pyro.sample(l, w_dist)


class SigmoidGuide(LinearModelGuide):

    def __init__(self, d, w_sizes, scale_tril_init=3., mu_delta_init=25., **kwargs):
        super(SigmoidGuide, self).__init__(d, w_sizes, scale_tril_init=scale_tril_init,
                                           **kwargs)
        self.mu0 = {l: nn.Parameter(
                -mu_delta_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu0 = nn.ParameterList(self.mu0.values())

        self.mu1 = {l: nn.Parameter(
                mu_delta_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu1 = nn.ParameterList(self.mu1.values())

        self.scale_tril0 = {l: nn.Parameter(
                scale_tril_init*torch.ones(*d, p, p)) for l, p in w_sizes.items()}
        self._registered0 = nn.ParameterList(self.scale_tril0.values())

        self.scale_tril1 = {l: nn.Parameter(
                scale_tril_init*torch.ones(*d, p, p)) for l, p in w_sizes.items()}
        self._registered1 = nn.ParameterList(self.scale_tril1.values())

        # TODO read from torch float specs
        self.epsilon = torch.tensor(2**-24)

    def get_params(self, y_dict, design, target_labels):

        # For values in (0, 1), we can perfectly invert the transformation
        y = torch.cat(list(y_dict.values()), dim=-1)
        mask0 = (y <= self.epsilon).squeeze(-1)
        mask1 = (1.-y <= self.epsilon).squeeze(-1)
        y, y1m = y.clamp(self.epsilon, 1), (1.-y).clamp(self.epsilon, 1)
        logited = y.log() - y1m.log()
        y_trans = logited

        mu, scale_tril = self.linear_model_formula(y_trans, design, target_labels)
        scale_tril = {l: scale_tril[l].expand(mu[l].shape + (mu[l].shape[-1], )) for l in scale_tril}

        # Now deal with clipping- values equal to 0 or 1
        for l in mu.keys():
            # print(self.mu1[l])
            # print(self.scale_tril1[l])
            mu[l][mask0, :] += self.mu0[l].expand(mu[l].shape)[mask0, :]
            mu[l][mask1, :] += self.mu1[l].expand(mu[l].shape)[mask1, :]
            scale_tril[l][mask0, :, :] += rtril(self.scale_tril0[l].expand(scale_tril[l].shape))[mask0, :, :]
            scale_tril[l][mask1, :, :] += rtril(self.scale_tril1[l].expand(scale_tril[l].shape))[mask1, :, :]

        return mu, scale_tril


class LogisticGuide(LinearModelGuide):

    def __init__(self, d, w_sizes, scale_tril_init=3., mu_init=0., **kwargs):
        super(LogisticGuide, self).__init__(d, w_sizes, scale_tril_init=scale_tril_init,
                                           **kwargs)
        self.mu0 = {l: nn.Parameter(
                mu_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu0 = nn.ParameterList(self.mu0.values())

        self.mu1 = {l: nn.Parameter(
                mu_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu1 = nn.ParameterList(self.mu1.values())

        self.scale_tril0 = {l: nn.Parameter(
                scale_tril_init*torch.ones(*d, p, p)) for l, p in w_sizes.items()}
        self._registered0 = nn.ParameterList(self.scale_tril0.values())

        self.scale_tril1 = {l: nn.Parameter(
                scale_tril_init*torch.ones(*d, p, p)) for l, p in w_sizes.items()}
        self._registered1 = nn.ParameterList(self.scale_tril1.values())


    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        mask0 = (y == 0.).squeeze(-1)
        mask1 = (y == 1.).squeeze(-1)

        mu = {}
        scale_tril = {}
        for l in target_labels:
            mu[l] = torch.empty(y.shape[:-1] + (self.w_sizes[l], ))
            scale_tril[l] = torch.empty(y.shape[:-1] + (self.w_sizes[l], self.w_sizes[l]))
            mu[l][mask0, :] = self.mu0[l].expand(mu[l].shape)[mask0, :]
            mu[l][mask1, :] = self.mu1[l].expand(mu[l].shape)[mask1, :]
            scale_tril[l][mask0, :, :] = rtril(self.scale_tril0[l].expand(scale_tril[l].shape))[mask0, :, :]
            scale_tril[l][mask1, :, :] = rtril(self.scale_tril1[l].expand(scale_tril[l].shape))[mask1, :, :]
        
        return mu, scale_tril


class SigmoidResponseEst(nn.Module):

    def __init__(self, d, observation_labels, mu_init=0., sigma_init=20., **kwargs):

        super(SigmoidResponseEst, self).__init__()

        self.mu = {l: nn.Parameter(mu_init*torch.ones(*d, 1)) for l in observation_labels}
        self.sigma = {l: nn.Parameter(sigma_init*torch.ones(*d, 1)) for l in observation_labels}
        self._registered_mu = nn.ParameterList(self.mu.values())
        self._registered_sigma = nn.ParameterList(self.sigma.values())
        # TODO read from torch float specs
        self.epsilon = torch.tensor(2**-24)
        self.softplus = nn.Softplus()

    def forward(self, design, observation_labels, target_labels):

        pyro.module("gibbs_y_guide", self)

        for l in observation_labels:
            self.sample_sigmoid(l, self.mu[l], self.sigma[l])

    def sample_sigmoid(self, label, mu, sigma):

            base_dist = dist.Normal(mu, self.softplus(sigma))
            tr_dist = dist.TransformedDistribution(base_dist, [SigmoidTransform()])
            response_dist = dist.CensoredDistribution(
                    tr_dist, upper_lim=1.-self.epsilon, lower_lim=self.epsilon).independent(1)
            pyro.sample(label, response_dist)


class NormalResponseEst(nn.Module):

    def __init__(self, d, observation_dimensions, mu_init=0., sigma_init=3., **kwargs):

        super(NormalResponseEst, self).__init__()

        self.mu = {l: nn.Parameter(mu_init*torch.ones(*d, p)) for l, p in observation_dimensions.items()}
        self.scale_tril = {l: nn.Parameter(sigma_init*torch.ones(*d, p, p)) for l, p in observation_dimensions.items()}
        self._registered_mu = nn.ParameterList(self.mu.values())
        self._registered_scale_tril = nn.ParameterList(self.scale_tril.values())

    def forward(self, design, observation_labels, target_labels):

        pyro.module("gibbs_y_guide", self)

        for l in observation_labels:
            pyro.sample(l, dist.MultivariateNormal(self.mu[l], scale_tril=rtril(self.scale_tril[l])))


class NormalLikelihoodEst(NormalResponseEst):

    def __init__(self, d, w_sizes, observation_dimensions, mu_init=0., sigma_init=3., **kwargs):

        super(NormalLikelihoodEst, self).__init__(d, observation_dimensions, mu_init, sigma_init, **kwargs)
        self.w_sizes = w_sizes

    def forward(self, theta_dict, design, observation_labels, target_labels):

        theta = torch.cat(list(theta_dict.values()), dim=-1)
        indices = get_indices(target_labels, self.w_sizes)
        subdesign = design[..., indices]
        centre = rmv(subdesign, theta)

        pyro.module("gibbs_y_re_guide", self)

        for l in observation_labels:
            pyro.sample(l, dist.MultivariateNormal(centre + self.mu[l], scale_tril=rtril(self.scale_tril[l])))


class SigmoidLikelihoodEst(SigmoidResponseEst):

    def __init__(self, d, w_sizes, observation_labels, mu_init=0., sigma_init=20., **kwargs):

        super(SigmoidLikelihoodEst, self).__init__(d, observation_labels, mu_init, sigma_init, **kwargs)
        self.w_sizes = w_sizes

    def forward(self, theta_dict, design, observation_labels, target_labels):

        theta = torch.cat(list(theta_dict.values()), dim=-1)
        indices = get_indices(target_labels, self.w_sizes)
        subdesign = design[..., indices]
        centre = rmv(subdesign, theta)
        
        pyro.module("gibbs_y_re_guide", self)

        for l in observation_labels:
            self.sample_sigmoid(l, centre + self.mu[l], self.sigma[l])


class LogisticResponseEst(nn.Module):
    
    def __init__(self, d, observation_labels, p_logit_init=0., **kwargs):

        super(LogisticResponseEst, self).__init__()

        self.logits = {l: nn.Parameter(p_logit_init*torch.ones(*d, 1)) for l in observation_labels}
        self._registered = nn.ParameterList(self.logits.values())

    
    def forward(self, design, observation_labels, target_labels):

        pyro.module("gibbs_y_guide", self)

        for l in observation_labels:
            y_dist = dist.Bernoulli(logits=self.logits[l]).independent(1)
            pyro.sample(l, y_dist)


class LogisticLikelihoodEst(nn.Module):
    
    def __init__(self, d, w_sizes, observation_labels, p_logit_init=0., **kwargs):

        super(LogisticLikelihoodEst, self).__init__()

        self.logit_correction = {l: nn.Parameter(p_logit_init*torch.ones(*d, 1)) for l in observation_labels}
        self.logit_offset = {l: nn.Parameter(torch.zeros(*d, 1)) for l in observation_labels}
        self._registered_correction = nn.ParameterList(self.logit_correction.values())
        self._registered_offset = nn.ParameterList(self.logit_offset.values())
        self.w_sizes = w_sizes
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
    def forward(self, theta_dict, design, observation_labels, target_labels):

        theta = torch.cat(list(theta_dict.values()), dim=-1)
        indices = get_indices(target_labels, self.w_sizes)
        subdesign = design[..., indices]
        centre = rmv(subdesign, theta)

        pyro.module("gibbs_y_re_guide", self)

        for l in observation_labels:
            p = .5*(self.sigmoid(centre + self.logit_offset[l] + self.softplus(self.logit_correction[l]))
                    + self.sigmoid(centre + self.logit_offset[l] - self.softplus(self.logit_correction[l])))
            y_dist = dist.Bernoulli(p).independent(1)
            pyro.sample(l, y_dist)


class NormalInverseGammaGuide(LinearModelGuide):

    def __init__(self, d, w_sizes, mf=False, tau_label="tau", alpha_init=10.,
                 b0_init=10., **kwargs):
        super(NormalInverseGammaGuide, self).__init__(d, w_sizes, **kwargs)
        self.alpha = nn.Parameter(alpha_init*torch.ones(*d))
        self.b0 = nn.Parameter(b0_init*torch.ones(*d))
        self.mf = mf
        self.tau_label = tau_label

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)

        coefficient_labels = [label for label in target_labels if label != self.tau_label]
        mu, scale_tril = self.linear_model_formula(y, design, coefficient_labels)
        mu_vec = torch.cat(list(mu.values()), dim=-1)

        yty = rvv(y, y)
        ytxmu = rvv(y, rmv(design, mu_vec))
        beta = self.b0 + .5*(yty - ytxmu)

        return mu, scale_tril, self.alpha, beta

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("ba_guide", self)

        mu, scale_tril, alpha, beta = self.get_params(y_dict, design, target_labels)

        if self.tau_label in target_labels:
            tau_dist = dist.Gamma(alpha, beta)
            tau = pyro.sample(self.tau_label, tau_dist)
            obs_sd = 1./tau.sqrt().unsqueeze(-1).unsqueeze(-1)

        for label in target_labels:
            if label != self.tau_label:
                if self.mf:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label])
                else:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label]*obs_sd)
                pyro.sample(label, w_dist)


class GuideDV(nn.Module):
    """A Donsker-Varadhan `T` family based on a guide family via
    the relation `T = log p(theta) - log q(theta | y, d)`
    """
    def __init__(self, guide):
        super(GuideDV, self).__init__()
        self.guide = guide

    def forward(self, design, trace, observation_labels, target_labels):

        trace.compute_log_prob()
        prior_lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        conditional_guide = pyro.condition(self.guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
                y_dict, design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        posterior_lp = sum(cond_trace.nodes[l]["log_prob"] for l in target_labels)

        return posterior_lp - prior_lp
