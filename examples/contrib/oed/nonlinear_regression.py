from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.util import iter_iaranges_to_shape

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


def gk_regression(centre_mean, centre_scale_tril, scale_alpha, scale_beta,
                  observation_sd, centre_label="centre", scale_label="scale", observation_label="y"):

    def model(design):
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for iarange in iter_iaranges_to_shape(batch_shape):
                stack.enter_context(iarange)
            centre_shape = batch_shape + (1, design.shape[-1])
            centre_st_shape = batch_shape + (1, design.shape[-1], design.shape[-1])
            centre_dist = dist.MultivariateNormal(centre_mean.expand(centre_shape),
                                                  scale_tril=centre_scale_tril.expand(centre_st_shape)).independent(1)
            centre = pyro.sample(centre_label, centre_dist)
            scale_dist = dist.Gamma(scale_alpha.expand(batch_shape), scale_beta.expand(batch_shape))
            scale = pyro.sample(scale_label, scale_dist).unsqueeze(-1)
            g = torch.exp(-(design - centre).pow(2).sum(-1)/scale)
            emission_dist = dist.Normal(g, observation_sd).independent(1)
            return pyro.sample(observation_label, emission_dist)

    model.observation_label = observation_label
    model.w_sizes = {}
    return model


def sinusoid_regression(amplitude_alpha, amplitude_beta, shift_mean, shift_sd, observation_sd, variable_noise=True,
                        amplitude_label="amplitude", shift_label="shift", observation_label="y"):

    def model(design):
        design = design.squeeze(-1)
        batch_shape = design.shape[:-1]
        with ExitStack() as stack:
            for iarange in iter_iaranges_to_shape(batch_shape):
                stack.enter_context(iarange)
            amplitude_dist = dist.Gamma(amplitude_alpha.expand(batch_shape), amplitude_beta.expand(batch_shape))
            amplitude = pyro.sample(amplitude_label, amplitude_dist).unsqueeze(-1)
            shift_dist = dist.Normal(shift_mean.expand(batch_shape), shift_sd.expand(batch_shape))
            shift = pyro.sample(shift_label, shift_dist).unsqueeze(-1)
            g = amplitude * torch.sin(shift + design)
            if variable_noise:
                emission_dist = dist.Normal(g, (1. + torch.abs(design)) * observation_sd).independent(1)
            else:
                emission_dist = dist.Normal(g, observation_sd).independent(1)
            return pyro.sample(observation_label, emission_dist)

    model.observation_label = observation_label
    model.w_sizes = {}
    return model
