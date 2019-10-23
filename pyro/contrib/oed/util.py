import math
import torch

import pyro
import pyro.poutine as poutine
from pyro.contrib.util import get_indices, lexpand, rexpand
from pyro.contrib.glmm import analytic_posterior_cov
from pyro.contrib.oed.eig import posterior_ape, vi_ape, laplace_vi_ape, xexpx
from pyro.contrib.autoguide import mean_field_entropy


def normal_inverse_gamma_ground_truth(model, design, observation_labels, target_labels, eig=True):
    lm = linear_model_ground_truth(model, design, observation_labels, target_labels, eig=eig)

    sign = 2.*(eig - .5)
    p = design.shape[-2]
    nu = model.alpha*2
    correction_factor = -(p/2.)*(math.log(2) + 1.) - torch.lgamma((nu+p)/2) + torch.lgamma(nu/2) \
        + (p/2.)*torch.log(nu) + ((nu+p)/2.)*(torch.digamma((nu+p)/2) - torch.digamma(nu/2))
    variance_factor = (p/2.)*(-torch.log(model.alpha) + torch.digamma(model.alpha))
    return lm + sign*(correction_factor + variance_factor)


def linear_model_ground_truth(model, design, observation_labels, target_labels, eig=True):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    design_shape = design.shape
    posterior_covs = [analytic_posterior_cov(prior_cov, x, model.obs_sd) for x in
                      torch.unbind(design.reshape(-1, design_shape[-2], design_shape[-1]))]
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_posterior_covs = [S[target_indices, :][:, target_indices] for S in posterior_covs]
    output = torch.tensor([0.5 * torch.logdet(2 * math.pi * math.e * C)
                           for C in target_posterior_covs])
    if eig:
        prior_entropy = mean_field_entropy(model, [design], whitelist=target_labels)
        output = prior_entropy - output

    return output.reshape(design.shape[:-2])


def logistic_extrapolation_ground_truth(model, design, observation_labels, target_labels, ythetaspace, num_samples=100):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_samples, 1)  # N copies of the model
    shape = list(design.shape[:-1])
    expanded_yspace = {k: rexpand(y, *shape) for k, y in ythetaspace.items()}
    newmodel = pyro.condition(model, data=expanded_yspace)
    trace = poutine.trace(newmodel).get_trace(expanded_design)
    trace.compute_log_prob()

    lpo = sum(trace.nodes[l]["log_prob"] for l in observation_labels)
    lpt = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    lpj = lpo + lpt

    # Joint
    first_term = xexpx(lpj.logsumexp(0) - math.log(num_samples)).sum(0)
    # Product of marginals
    second_term = xexpx(lpo.logsumexp(0) - math.log(num_samples)).sum(0)/2 + xexpx(
        lpt.logsumexp(0) - math.log(num_samples)).sum(0)/2

    return first_term - second_term


def lm_H_prior(model, design, observation_labels, target_labels):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_prior_covs = prior_cov[target_indices, :][:, target_indices]
    return 0.5*torch.logdet(2 * math.pi * math.e * target_prior_covs)


def mc_H_prior(model, design, observation_labels, target_labels, num_samples=1000):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_samples)
    trace = pyro.poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    return -lp.sum(0)/num_samples


def extrap_H_prior(model, design, observation_labels, target_labels, num_samples=10000):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_samples)
    cm = pyro.condition(model, {"y": torch.ones(expanded_design.shape[:-1])})
    trace = pyro.poutine.trace(cm).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    log_p_hat = lp.logsumexp(0) - math.log(num_samples)
    log_1mp = torch.log(1. - torch.exp(log_p_hat))
    return -xexpx(log_p_hat) - xexpx(log_1mp)


def vi_eig_lm(model, design, observation_labels, target_labels, *args, **kwargs):
    # **Only** applies to linear models - analytic prior entropy
    ape = vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
    return prior_entropy - ape


def vi_eig_mc(model, design, observation_labels, target_labels, *args, **kwargs):
    # Compute the prior entropy by Monte Carlo, then uses vi_ape
    if "num_hprior_samples" in kwargs:
        hprior = mc_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = mc_H_prior(model, design, observation_labels, target_labels)
    ape = vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    return hprior - ape


def laplace_vi_eig_mc(model, design, observation_labels, target_labels, *args, **kwargs):
    # Compute the prior entropy by Monte Carlo, then uses vi_ape
    if "num_hprior_samples" in kwargs:
        hprior = mc_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = mc_H_prior(model, design, observation_labels, target_labels)
    ape = laplace_vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    return hprior - ape


def posterior_eig_lm(model, design, observation_labels, target_labels, *args, **kwargs):
    # **Only** applies to linear models - analytic prior entropy
    ape = posterior_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
    if isinstance(ape, tuple):
        return tuple(prior_entropy - a for a in ape)
    else:
        return prior_entropy - ape


def posterior_eig_mc(model, design, observation_labels, target_labels, *args, **kwargs):
    # Compute the prior entropy my Monte Carlo, then uses posterior_ape
    if "num_hprior_samples" in kwargs:
        hprior = mc_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = mc_H_prior(model, design, observation_labels, target_labels)
    return hprior - posterior_ape(model, design, observation_labels, target_labels, *args, **kwargs)


def posterior_eig_extrap(model, design, observation_labels, target_labels, *args, **kwargs):
    # Compute the prior entropy my Monte Carlo, then uses posterior_ape. For extrapolation
    if "num_hprior_samples" in kwargs:
        hprior = extrap_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = extrap_H_prior(model, design, observation_labels, target_labels)
    return hprior - posterior_ape(model, design, observation_labels, target_labels, *args, **kwargs)
