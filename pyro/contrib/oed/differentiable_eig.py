import torch
import math
import warnings

import pyro
from pyro import poutine
from pyro.contrib.autoguide import mean_field_entropy
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI
from pyro.util import torch_isnan, torch_isinf
from pyro.contrib.util import lexpand
from pyro.contrib.oed.eig import _safe_mean_terms


def _differentiable_posterior_loss(model, guide, observation_labels, target_labels):
    """This version of the loss function deals with the case that `y` is not reparametrizable."""

    def loss_fn(design, num_particles, control_variate=0., **kwargs):

        expanded_design = lexpand(design, num_particles)

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        # Run through q(theta | y, d)
        conditional_guide = pyro.condition(guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        terms = -sum(cond_trace.nodes[l]["log_prob"] for l in target_labels)

        # Calculate the score parts
        trace.compute_score_parts()
        prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)
        terms += (terms.detach() - control_variate) * prescore_function

        return _safe_mean_terms(terms)

    return loss_fn
