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
        #print("score", trace.nodes["y"]["score_parts"][1])
        prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)
        terms += (terms.detach() - control_variate) * prescore_function

        return _safe_mean_terms(terms)

    return loss_fn


def differentiable_nce_eig(model, design, observation_labels, target_labels=None, N=100, M=10, control_variate=0.,
                           **kwargs):

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    trace = poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)

    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, N)  # sample M theta
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    marginal_log_probs = torch.cat([lexpand(conditional_lp, 1),
                                    sum(retrace.nodes[l]["log_prob"] for l in observation_labels)])
    marginal_lp = marginal_log_probs.logsumexp(0) - math.log(M+1)

    terms = conditional_lp - marginal_lp

    # Calculate the score parts
    trace.compute_score_parts()
    prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)
    terms += (terms.detach() - control_variate) * prescore_function

    return _safe_mean_terms(terms)


def _differentiable_ace_eig_loss(model, guide, M, observation_labels, target_labels):

    def loss_fn(design, num_particles, control_variate=0., **kwargs):
        N = num_particles
        expanded_design = lexpand(design, N)

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict_exp = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        trace.compute_log_prob()
        marginal_terms_cross = sum(trace.nodes[l]["log_prob"] for l in target_labels)
        marginal_terms_cross += sum(trace.nodes[l]["log_prob"] for l in observation_labels)

        reguide_trace = poutine.trace(pyro.condition(guide, data=theta_dict)).get_trace(
            y_dict, expanded_design, observation_labels, target_labels
        )
        reguide_trace.compute_log_prob()
        marginal_terms_cross -= sum(reguide_trace.nodes[l]["log_prob"] for l in target_labels)

        # Sample M times from q(theta | y, d) for each y
        reexpanded_design = lexpand(expanded_design, M)
        guide_trace = poutine.trace(guide).get_trace(
            y_dict_exp, reexpanded_design, observation_labels, target_labels
        )
        theta_y_dict = {l: guide_trace.nodes[l]["value"] for l in target_labels}
        theta_y_dict.update(y_dict_exp)
        guide_trace.compute_log_prob()

        # Re-run that through the model to compute the joint
        model_trace = poutine.trace(pyro.condition(model, data=theta_y_dict)).get_trace(reexpanded_design)
        model_trace.compute_log_prob()

        marginal_terms_proposal = -sum(guide_trace.nodes[l]["log_prob"] for l in target_labels)
        marginal_terms_proposal += sum(model_trace.nodes[l]["log_prob"] for l in target_labels)
        marginal_terms_proposal += sum(model_trace.nodes[l]["log_prob"] for l in observation_labels)

        marginal_terms = torch.cat([lexpand(marginal_terms_cross, 1), marginal_terms_proposal])
        terms = -marginal_terms.logsumexp(0) + math.log(M + 1)

        terms += sum(trace.nodes[l]["log_prob"] for l in observation_labels)

        # Calculate the score parts
        trace.compute_score_parts()
        prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)

        # This is necessary for discrete theta
        # guide_trace.compute_score_parts()
        # guide_score_component = sum(guide_trace.nodes[l]["score_parts"][1] for l in target_labels)
        # if not isinstance(guide_score_component, int):
        #     guide_score_component = guide_score_component.sum(0)
        # prescore_function += guide_score_component

        terms += (terms.detach() - control_variate) * prescore_function

        return _safe_mean_terms(terms)

    return loss_fn