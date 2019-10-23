from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.util import rmv, rexpand, lexpand, rtril, iter_plates_to_shape
from pyro.contrib.glmm import broadcast_cat
from .turk_experiment import gen_design_space, design_matrix

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2

EPSILON = torch.tensor(2 ** -24)


CANDIDATE_DESIGNS = gen_design_space()
matrices = [design_matrix(d, 1, 1) for d in CANDIDATE_DESIGNS]
matrices.sort(key=lambda x: x.abs().sum())
turk_designs = torch.stack(matrices, dim=0)

p_f = p_re = 6
num_parallel = 1
num_participants = 1


def turk_model():

    def sample_latents(full_design, div):
        design, slope_design = full_design[..., :div], full_design[..., div:]
        # design is size batch x n x p
        n, p = design.shape[-2:]

        model_fixed_effect_mean = lexpand(torch.zeros(p_f), num_parallel, 1)
        model_fixed_effect_scale_tril = 10. * lexpand(torch.eye(p_f), num_parallel, 1)
        model_random_effect_mean = lexpand(torch.zeros(p_re), num_parallel, 1)
        model_random_effect_precision_alpha = lexpand(torch.tensor(2.), num_parallel, 1)
        model_random_effect_precision_beta = lexpand(torch.tensor(2.), num_parallel, 1)
        model_slope_precision_alpha = lexpand(torch.tensor(2.), num_parallel, 1)
        model_slope_precision_beta = lexpand(torch.tensor(2.), num_parallel, 1)

        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            # Build the regression coefficient
            w = []

            ###############
            # Fixed effects
            ###############
            fixed_effect_mean = model_fixed_effect_mean
            fixed_effect_dist = dist.MultivariateNormal(
                fixed_effect_mean.expand(batch_shape + (fixed_effect_mean.shape[-1],)),
                scale_tril=rtril(model_fixed_effect_scale_tril))
            w.append(pyro.sample("fixed_effects", fixed_effect_dist))

            ################
            # Random effects
            ################
            re_precision_dist = dist.Gamma(
                model_random_effect_precision_alpha.expand(batch_shape),
                model_random_effect_precision_beta.expand(batch_shape))
            re_precision = pyro.sample("random_effects_precision", re_precision_dist)
            # Sample a fresh sd for each batch, re-use it for each random effect
            re_mean = model_random_effect_mean
            re_sd = rexpand(1./torch.sqrt(re_precision), re_mean.shape[-1])
            re_dist = dist.Normal(re_mean.expand(batch_shape + (re_mean.shape[-1],)), re_sd).to_event(1)
            w.append(pyro.sample("random_effects", re_dist))

            # Regression coefficient `w` is batch x p
            w = broadcast_cat(w)

            ##############
            # Random slope
            ##############
            slope_precision_dist = dist.Gamma(
                model_slope_precision_alpha.expand(batch_shape),
                model_slope_precision_beta.expand(batch_shape))
            slope_precision = pyro.sample("random_slope_precision", slope_precision_dist)
            slope_sd = rexpand(1./torch.sqrt(slope_precision), slope_design.shape[-1])
            slope_dist = dist.LogNormal(0., slope_sd).to_event(1)
            slope = rmv(slope_design, pyro.sample("random_slope", slope_dist).clamp(1e-5, 1e5))

            return w, slope

    def sample_emission(full_design, w, slope, div):
        model_obs_sd = lexpand(torch.tensor(10.), num_parallel, 1)

        design = full_design[..., :div]
        batch_shape = design.shape[:-2]
        obs_sd = model_obs_sd.expand(batch_shape).unsqueeze(-1)

        ###################
        # Sigmoid transform
        ###################
        # Run the regressor forward conditioned on inputs
        prediction_mean = rmv(design, w)
        response_dist = dist.CensoredSigmoidNormal(
            loc=slope*prediction_mean, scale=slope*obs_sd, upper_lim=1.-EPSILON, lower_lim=EPSILON
        ).to_event(1)
        return pyro.sample("y", response_dist)

    def model(design):

        w, slope = sample_latents(design, div=p_f+p_re)
        return sample_emission(design, w, slope, div=p_f+p_re)

    model.w_sizes = OrderedDict([("fixed_effects", p_f), ("random_effects", p_f)])
    model.observation_label = "y"
    return model
