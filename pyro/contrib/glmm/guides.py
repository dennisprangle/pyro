from __future__ import absolute_import, division, print_function

import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.util import (
    get_indices, tensor_to_dict, rmv, rvv, rtril, rdiag, lexpand, rexpand, iter_iaranges_to_shape
)
from pyro.ops.linalg import rinverse
from pyro.util import is_bad

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


class LinearModelPosteriorGuide(nn.Module):

    def __init__(self, d, w_sizes, tikhonov_init=-2., scale_tril_init=3., **kwargs):
        """
        Guide for linear models. No amortisation happens over designs.
        Amortisation over data is taken care of by analytic formulae for
        linear models (heavy use of truth).

        :param tuple d: the number of designs
        :param dict w_sizes: map from variable string names to int.
        :param float tikhonov_init: initial value for `tikhonov_diag` parameter.
        :param float scale_tril_init: initial value for `scale_tril` parameter.
        """
        super(LinearModelPosteriorGuide, self).__init__()
        # Represent each parameter group as independent Gaussian
        # Making a weak mean-field assumption
        # To avoid this- combine labels
        if not isinstance(d, (tuple, list, torch.Tensor)):
            d = (d,)
        self.tikhonov_diag = nn.Parameter(
                tikhonov_init*torch.ones(*d, sum(w_sizes.values())))
        self.scaled_prior_mean = nn.Parameter(torch.zeros(*d, sum(w_sizes.values())))
        self.scale_tril = {l: nn.Parameter(
                scale_tril_init*lexpand(torch.eye(p), *d)) for l, p in w_sizes.items()}
        # This registers the dict values in pytorch
        # Await new version to use nn.ParamterDict
        self._registered = nn.ParameterList(self.scale_tril.values())
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        return self.linear_model_formula(y, design, target_labels)

    def linear_model_formula(self, y, design, target_labels):

        tikhonov_diag = rdiag(torch.exp(self.tikhonov_diag))
        xtx = torch.matmul(design.transpose(-1, -2), design) + tikhonov_diag
        xtxi = rinverse(xtx, sym=True)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y) + self.scaled_prior_mean)

        # Extract sub-indices
        mu = tensor_to_dict(self.w_sizes, mu, subset=target_labels)
        scale_tril = {l: rtril(self.scale_tril[l]) for l in target_labels}

        return mu, scale_tril

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("posterior_guide", self)

        # Returns two dicts from labels -> tensors
        mu, scale_tril = self.get_params(y_dict, design, target_labels)

        for l in target_labels:
            w_dist = dist.MultivariateNormal(mu[l], scale_tril=scale_tril[l])
            pyro.sample(l, w_dist)


class LinearModelLaplaceGuide(nn.Module):
    """
    Laplace approximation for a (G)LM
    """
    def __init__(self, d, w_sizes, tau_label=None, init_value=0.1, **kwargs):
        super(LinearModelLaplaceGuide, self).__init__()
        # start in train mode
        self.train()
        if not isinstance(d, (tuple, list, torch.Tensor)):
            d = (d,)
        self.means = {}
        if tau_label is not None:
            w_sizes[tau_label] = 1
        for l, mu_l in tensor_to_dict(w_sizes, init_value*torch.ones(*d, sum(w_sizes.values()))).items():
            self.means[l] = nn.Parameter(mu_l)
        self._registered = nn.ParameterList(self.means.values())
        self.scale_trils = {}
        self.w_sizes = w_sizes

    @staticmethod
    def _hessian_diag(y, x, event_shape):
        batch_shape = x.shape[:-len(event_shape)]
        assert tuple(x.shape) == tuple(batch_shape) + tuple(event_shape)

        dy = torch.autograd.grad(y, [x,], create_graph=True)[0]
        H = []

        # collapse independent dimensions into a single one,
        # and dependent dimensions into another single one
        batch_size = 1
        for batch_shape_dim in batch_shape:
            batch_size *= batch_shape_dim

        event_size = 1
        for event_shape_dim in event_shape:
            event_size *= event_shape_dim

        flat_dy = dy.reshape(batch_size, event_size)

        # loop over dependent part
        for i in range(flat_dy.shape[-1]):
            dyi = flat_dy.index_select(-1, torch.tensor([i]))
            Hi = torch.autograd.grad([dyi], [x,], grad_outputs=[torch.ones_like(dyi)], retain_graph=True)[0]
            H.append(Hi)
        H = torch.stack(H, -1).reshape(*(x.shape + event_shape))
        return H

    @staticmethod
    def _batch_potrf_compat(x, **kwargs):
        # TODO update this code to Pyro dev and remove in favor of .cholesky()
        flat_x = x.reshape(-1, x.shape[-2], x.shape[-1])
        potrfs = torch.stack([flat_xi.potrf(**kwargs) for flat_xi in flat_x])
        return potrfs.reshape(*tuple(x.shape))

    def finalize(self, loss, target_labels):
        """
        Compute the Hessian of the parameters wrt ``loss``

        Usage::

            >>> guide = LinearModelLaplace(...)
            >>> guide.finalize(loss)
        """
        # set self.training = False
        self.eval()
        for l, mu_l in self.means.items():
            if l not in target_labels:
                continue
            hess_l = self._hessian_diag(loss, mu_l, event_shape=(self.w_sizes[l],))
            cov_l = rinverse(hess_l)
            self.scale_trils[l] = rtril(self._batch_potrf_compat(cov_l, upper=False))

    def forward(self, design, target_labels=None):
        """
        Sample the posterior
        """
        if target_labels is None:
            target_labels = list(self.means.keys())

        pyro.module("laplace_guide", self)
        with ExitStack() as stack:
            stack.enter_context(poutine.broadcast())
            for iarange in iter_iaranges_to_shape(design.shape[:-2]):
                stack.enter_context(iarange)

            if self.training:
                # MAP via Delta guide
                for l in target_labels:
                    w_dist = dist.Delta(self.means[l]).independent(1)
                    pyro.sample(l, w_dist)
            else:
                # Laplace approximation via MVN with hessian
                for l in target_labels:
                    w_dist = dist.MultivariateNormal(self.means[l], scale_tril=self.scale_trils[l])
                    pyro.sample(l, w_dist)


# Deprecated
class SigmoidPosteriorGuide(LinearModelPosteriorGuide):

    def __init__(self, d, w_sizes, scale_tril_init=3., mu_init=0., **kwargs):
        super(SigmoidPosteriorGuide, self).__init__(d, w_sizes, scale_tril_init=scale_tril_init,
                                                    **kwargs)
        self.mu0 = {l: nn.Parameter(
                mu_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu0 = nn.ParameterList(self.mu0.values())

        self.mu1 = {l: nn.Parameter(
                mu_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu1 = nn.ParameterList(self.mu1.values())

        self.scale_tril0 = {l: nn.Parameter(
                scale_tril_init*lexpand(torch.eye(p), *d)) for l, p in w_sizes.items()}
        self._registered0 = nn.ParameterList(self.scale_tril0.values())

        self.scale_tril1 = {l: nn.Parameter(
                scale_tril_init*lexpand(torch.eye(p), *d)) for l, p in w_sizes.items()}
        self._registered1 = nn.ParameterList(self.scale_tril1.values())

        # TODO read from torch float specs
        self.epsilon = torch.tensor(2 ** -24)

    def get_params(self, y_dict, design, target_labels):

        # For values in (0, 1), we can perfectly invert the transformation
        y = torch.cat(list(y_dict.values()), dim=-1)
        mask0 = (y <= self.epsilon).squeeze(-1)
        mask1 = (1. - y <= self.epsilon).squeeze(-1)
        y_trans = y.log() - (1.-y).log()

        mu, scale_tril = self.linear_model_formula(y_trans, design, target_labels)
        scale_tril = {l: scale_tril[l].expand(mu[l].shape + (mu[l].shape[-1], )) for l in scale_tril}

        for l in mu.keys():
            mu[l] = mu[l] + (-mu[l] + self.mu0[l]) * rexpand(mask0.float(), 1)
            mu[l] = mu[l] + (-mu[l] + self.mu1[l]) * rexpand(mask1.float(), 1)
            scale_tril[l] = scale_tril[l] + (-scale_tril[l] + self.scale_tril0[l]) * rexpand(mask0.float(), 1, 1)
            scale_tril[l] = scale_tril[l] + (-scale_tril[l] + self.scale_tril1[l]) * rexpand(mask1.float(), 1, 1)
            scale_tril[l] = rtril(scale_tril[l])

        return mu, scale_tril


class SigmoidLocationPosteriorGuide(nn.Module):

    def __init__(self, d, w_sizes, scale_tril_init, multiplier, prior_mean, **kwargs):
        assert len(w_sizes) == 1
        super(SigmoidLocationPosteriorGuide, self).__init__()
        self.label, p = list(w_sizes.items())[0]

        self.multiplier = multiplier
        self.weights = nn.Parameter(torch.zeros(*d, p))
        self.mean_offsets = nn.ParameterList([nn.Parameter(torch.zeros(*d, p)), nn.Parameter(torch.zeros(*d, p))])
        self.scale_trils = nn.ParameterList([nn.Parameter(scale_tril_init * lexpand(torch.eye(p), *d)),
                                             nn.Parameter(scale_tril_init * lexpand(torch.eye(p), *d)),
                                             nn.Parameter(scale_tril_init * lexpand(torch.eye(p), *d))])
        self.prior_mean = prior_mean

        # TODO read from torch float specs
        self.epsilon = torch.tensor(2 ** -24)

    def forward(self, y_dict, design, observation_labels, target_labels):
        test_point = rvv(design, self.multiplier)
        # For values in (0, 1), we can perfectly invert the transformation
        y = torch.cat(list(y_dict.values()), dim=-1)
        mask0 = (y <= self.epsilon).squeeze(-1)
        mask1 = (1. - y <= self.epsilon).squeeze(-1)
        y_trans = y.log() - (1. - y).log()
        eta = test_point - y_trans

        mu = self.weights * eta + (1 - self.weights) * self.prior_mean
        scale_tril = self.scale_trils[1]

        mu = mu + self.mean_offsets[0] * rexpand(mask0.float(), 1) + self.mean_offsets[1] * rexpand(mask1.float(), 1)
        scale_tril = scale_tril + (-scale_tril + self.scale_trils[0]) * rexpand(mask0.float(), 1, 1)
        scale_tril = scale_tril + (-scale_tril + self.scale_trils[2]) * rexpand(mask1.float(), 1, 1)
        scale_tril = rtril(scale_tril)

        pyro.module("posterior_guide", self)

        w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        pyro.sample(self.label, w_dist)


class LogisticPosteriorGuide(LinearModelPosteriorGuide):

    def __init__(self, d, w_sizes, scale_tril_init=3., mu_init=0., **kwargs):
        super(LogisticPosteriorGuide, self).__init__(d, w_sizes, scale_tril_init=scale_tril_init,
                                                     **kwargs)
        self.mu0 = {l: nn.Parameter(
                mu_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu0 = nn.ParameterList(self.mu0.values())

        self.mu1 = {l: nn.Parameter(
                mu_init*torch.ones(*d, p)) for l, p in w_sizes.items()}
        self._registered_mu1 = nn.ParameterList(self.mu1.values())

        self.scale_tril0 = {l: nn.Parameter(
                scale_tril_init*lexpand(torch.eye(p), *d)) for l, p in w_sizes.items()}
        self._registered0 = nn.ParameterList(self.scale_tril0.values())

        self.scale_tril1 = {l: nn.Parameter(
                scale_tril_init*lexpand(torch.eye(p), *d)) for l, p in w_sizes.items()}
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


class NormalInverseGammaPosteriorGuide(LinearModelPosteriorGuide):

    def __init__(self, d, w_sizes, mf=False, correct_gamma=False, tau_label="tau", alpha_init=10.,
                 b0_init=10., **kwargs):
        super(NormalInverseGammaPosteriorGuide, self).__init__(d, w_sizes, **kwargs)
        self.alpha = nn.Parameter(alpha_init*torch.ones(*d))
        self.b0 = nn.Parameter(b0_init*torch.ones(*d))
        self.mf = mf
        self.correct_gamma = correct_gamma
        self.tau_label = tau_label

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)

        coefficient_labels = [label for label in target_labels if label != self.tau_label]
        mu, scale_tril = self.linear_model_formula(y, design, coefficient_labels)

        if self.correct_gamma:
            mu_vec = torch.cat(list(mu.values()), dim=-1)
            yty = rvv(y, y)
            ytxmu = rvv(y, rmv(design, mu_vec))
            beta = self.b0 + .5*(yty - ytxmu)
        else:
            # Treat beta as a constant
            beta = self.b0

        return mu, scale_tril, self.alpha, beta

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("posterior_guide", self)

        mu, scale_tril, alpha, beta = self.get_params(y_dict, design, target_labels)

        if self.tau_label in target_labels:
            tau_dist = dist.Gamma(alpha.unsqueeze(-1), beta.unsqueeze(-1)).independent(1)
            tau = pyro.sample(self.tau_label, tau_dist)
            obs_sd = 1./tau.sqrt().unsqueeze(-1)

        for label in target_labels:
            if label != self.tau_label:
                if self.mf:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label])
                else:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label]*obs_sd)
                pyro.sample(label, w_dist)


class NormalMarginalGuide(nn.Module):

    def __init__(self, d, y_sizes, mu_init=0., sigma_init=3., **kwargs):

        super(NormalMarginalGuide, self).__init__()

        self.mu = {l: nn.Parameter(mu_init*torch.ones(*d, p)) for l, p in y_sizes.items()}
        self.scale_tril = {l: nn.Parameter(sigma_init*lexpand(torch.eye(p), *d)) for l, p in y_sizes.items()}
        self._registered_mu = nn.ParameterList(self.mu.values())
        self._registered_scale_tril = nn.ParameterList(self.scale_tril.values())

    def forward(self, design, observation_labels, target_labels):

        pyro.module("marginal_guide", self)

        for l in observation_labels:
            pyro.sample(l, dist.MultivariateNormal(self.mu[l], scale_tril=rtril(self.scale_tril[l])))


class NormalLikelihoodGuide(NormalMarginalGuide):

    def __init__(self, d, w_sizes, y_sizes, mu_init=0., sigma_init=3., **kwargs):

        super(NormalLikelihoodGuide, self).__init__(d, y_sizes, mu_init, sigma_init, **kwargs)
        self.w_sizes = w_sizes

    def forward(self, theta_dict, design, observation_labels, target_labels):

        theta = torch.cat(list(theta_dict.values()), dim=-1)
        indices = get_indices(target_labels, self.w_sizes)
        subdesign = design[..., indices]
        centre = rmv(subdesign, theta)

        pyro.module("likelihood_guide", self)

        for l in observation_labels:
            pyro.sample(l, dist.MultivariateNormal(centre + self.mu[l], scale_tril=rtril(self.scale_tril[l])))


class SigmoidMarginalGuide(nn.Module):

    def __init__(self, d, y_sizes, mu_init=0., sigma_init=20., **kwargs):

        super(SigmoidMarginalGuide, self).__init__()

        assert all(d == 1 for d in y_sizes.values())
        self.mu = {l: nn.Parameter(mu_init*torch.ones(*d, 1)) for l in y_sizes}
        self.sigma = {l: nn.Parameter(sigma_init*torch.ones(*d, 1)) for l in y_sizes}
        self._registered_mu = nn.ParameterList(self.mu.values())
        self._registered_sigma = nn.ParameterList(self.sigma.values())
        # TODO read from torch float specs
        self.epsilon = torch.tensor(2**-24)
        self.softplus = nn.Softplus()

    def forward(self, design, observation_labels, target_labels):

        pyro.module("marginal_guide", self)

        for l in observation_labels:
            if is_bad(self.mu[l]):
                raise ArithmeticError("NaN in marginal mean")
            elif is_bad(self.sigma[l]):
                raise ArithmeticError("NaN in marginal sigma")
            self.sample_sigmoid(l, self.mu[l], self.sigma[l])

    def sample_sigmoid(self, label, mu, sigma):

            response_dist = dist.CensoredSigmoidNormal(
                loc=mu, scale=self.softplus(sigma), upper_lim=1.-self.epsilon, lower_lim=self.epsilon
            ).independent(1)
            pyro.sample(label, response_dist)


class SigmoidLikelihoodGuide(SigmoidMarginalGuide):

    def __init__(self, d, w_sizes, y_sizes, mu_init=0., sigma_init=10., **kwargs):

        super(SigmoidLikelihoodGuide, self).__init__(d, y_sizes, mu_init, sigma_init, **kwargs)
        self.log_multiplier = nn.Parameter(torch.zeros(*d, 1))
        self.w_sizes = w_sizes

    def forward(self, theta_dict, design, observation_labels, target_labels):

        theta = torch.cat(list(theta_dict.values()), dim=-1)
        indices = get_indices(target_labels, self.w_sizes)
        subdesign = design[..., indices]
        centre = rmv(subdesign, theta)
        scaled_centre = torch.exp(self.log_multiplier)*centre

        pyro.module("likelihood_guide", self)

        for l in observation_labels:
            if is_bad(self.mu[l]):
                raise ArithmeticError("NaN in likelihood mean")
            elif is_bad(self.sigma[l]):
                raise ArithmeticError("NaN in likelihood sigma")
            self.sample_sigmoid(l, scaled_centre + self.mu[l], self.sigma[l])


class LogisticMarginalGuide(nn.Module):
    
    def __init__(self, d, y_sizes, p_logit_init=0., **kwargs):

        super(LogisticMarginalGuide, self).__init__()

        self.logits = {l: nn.Parameter(p_logit_init*torch.ones(*d, 1)) for l in y_sizes}
        self._registered = nn.ParameterList(self.logits.values())

    def forward(self, design, observation_labels, target_labels):

        pyro.module("marginal_guide", self)

        for l in observation_labels:
            y_dist = dist.Bernoulli(logits=self.logits[l]).independent(1)
            pyro.sample(l, y_dist)


class LogisticLikelihoodGuide(nn.Module):
    
    def __init__(self, d, w_sizes, y_sizes, p_logit_init=0., **kwargs):

        super(LogisticLikelihoodGuide, self).__init__()

        self.logit_correction = {l: nn.Parameter(p_logit_init*torch.ones(*d, 1)) for l in y_sizes}
        self.logit_offset = {l: nn.Parameter(torch.zeros(*d, 1)) for l in y_sizes}
        self._registered_correction = nn.ParameterList(self.logit_correction.values())
        self._registered_offset = nn.ParameterList(self.logit_offset.values())
        self.log_multiplier = nn.Parameter(torch.zeros(*d, 1))
        self.w_sizes = w_sizes
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
    def forward(self, theta_dict, design, observation_labels, target_labels):

        theta = torch.cat(list(theta_dict.values()), dim=-1)
        indices = get_indices(target_labels, self.w_sizes)
        subdesign = design[..., indices]
        centre = rmv(subdesign, theta)
        scaled_centre = torch.exp(self.log_multiplier) * centre

        pyro.module("likelihood_guide", self)

        for l in observation_labels:
            p = .5*(self.sigmoid(scaled_centre + self.logit_offset[l] + self.softplus(self.logit_correction[l]))
                    + self.sigmoid(scaled_centre + self.logit_offset[l] - self.softplus(self.logit_correction[l])))
            y_dist = dist.Bernoulli(p).independent(1)
            pyro.sample(l, y_dist)


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
