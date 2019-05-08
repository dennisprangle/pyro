from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import torch

import pyro
from pyro.contrib.util import lexpand
from .turk_experiment import NewParticipantModel, gen_design_space, design_matrix


CANDIDATE_DESIGNS = gen_design_space()
matrices = [design_matrix(d, 1, 1) for d in CANDIDATE_DESIGNS]
matrices.sort(key=lambda x: x.abs().sum())
turk_designs = torch.stack(matrices, dim=0)

p = p_re = 6
num_parallel = 1
num_participants = 1


def turk_model():
    def model(*args, **kwargs):
        with pyro.poutine.block(hide_fn=lambda s: s["name"].startswith("model")):
            pyro.param("model_fixed_effect_mean", lexpand(torch.zeros(p), num_parallel, 1))
            pyro.param("model_fixed_effect_scale_tril", 10. * lexpand(torch.eye(p), num_parallel, 1))
            pyro.param("model_random_effect_mean", lexpand(torch.zeros(p_re), num_parallel, 1))
            pyro.param("model_random_effect_precision_alpha", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("model_random_effect_precision_beta", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("model_slope_precision_alpha", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("model_slope_precision_beta", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("model_obs_sd", lexpand(torch.tensor(10.), num_parallel, 1))
            pyro.param("model_random_effect_scale_tril", lexpand(torch.eye(p_re, p_re), num_parallel, 1))
            pyro.param("model_slope_mean", lexpand(torch.zeros(num_participants), num_parallel, 1))
            pyro.param("model_slope_sd", lexpand(4. * torch.ones(num_participants), num_parallel, 1))
            pyro.param("model_mixing_matrix", lexpand(torch.zeros(p_re, p), num_parallel, 1))
            return NewParticipantModel(
                "model_", p+p_re, hide_fn=lambda s: s["name"].startswith("model")).model(*args, **kwargs)
    model.w_sizes = OrderedDict([("fixed_effects", p), ("random_effects", p)])
    model.observation_label = "y"
    return model
