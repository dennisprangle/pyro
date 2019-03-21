from __future__ import absolute_import, division, print_function

import argparse
import itertools
import logging
import datetime
import pickle

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro import poutine
from pyro.contrib.util import rmv, rexpand, lexpand, rtril, iter_plates_to_shape
from pyro.contrib.glmm import broadcast_cat
from pyro.contrib.oed.eig import naive_rainforth_eig, gibbs_y_eig, gibbs_y_re_eig, elbo_learn
from pyro.contrib.glmm.guides import SigmoidMarginalGuide, SigmoidLikelihoodGuide

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2

EPSILON = torch.tensor(2 ** -24)
CHECK_RAINFORTH = False
SRC = './run_outputs/turk_simulation/0run_6.result_stream.pickle'


class NewParticipantModel:

    def __init__(self, prefix, div, **block_kwargs):
        # Note: the random effect shape is inferred from `random_effect_mean`
        # which should be expanded to the right shape
        self.prefix = prefix
        self.div = div
        self.block_kwargs = block_kwargs

    def sample_latents(self, full_design):
        design, slope_design = full_design[..., :self.div], full_design[..., self.div:]
        # design is size batch x n x p
        n, p = design.shape[-2:]
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            # Build the regression coefficient
            w = []

            ###############
            # Fixed effects
            ###############
            fixed_effect_mean = pyro.param(self.prefix+"fixed_effect_mean")
            fixed_effect_dist = dist.MultivariateNormal(
                fixed_effect_mean.expand(batch_shape + (fixed_effect_mean.shape[-1],)),
                scale_tril=rtril(pyro.param(self.prefix+"fixed_effect_scale_tril")))
            w.append(pyro.sample("fixed_effects", fixed_effect_dist))

            ################
            # Random effects
            ################
            re_precision_dist = dist.Gamma(
                pyro.param(self.prefix+"random_effect_precision_alpha").expand(batch_shape),
                pyro.param(self.prefix+"random_effect_precision_beta").expand(batch_shape))
            re_precision = pyro.sample("random_effects_precision", re_precision_dist)
            # Sample a fresh sd for each batch, re-use it for each random effect
            re_mean = pyro.param(self.prefix+"random_effect_mean")
            re_sd = rexpand(1./torch.sqrt(re_precision), re_mean.shape[-1])
            re_dist = dist.Normal(re_mean.expand(batch_shape + (re_mean.shape[-1],)), re_sd).to_event(1)
            w.append(pyro.sample("random_effects", re_dist))

            # Regression coefficient `w` is batch x p
            w = broadcast_cat(w)

            ##############
            # Random slope
            ##############
            slope_precision_dist = dist.Gamma(
                pyro.param(self.prefix+"slope_precision_alpha").expand(batch_shape),
                pyro.param(self.prefix+"slope_precision_beta").expand(batch_shape))
            slope_precision = pyro.sample("random_slope_precision", slope_precision_dist)
            slope_sd = rexpand(1./torch.sqrt(slope_precision), slope_design.shape[-1])
            slope_dist = dist.LogNormal(0., slope_sd).to_event(1)
            slope = rmv(slope_design, pyro.sample("random_slope", slope_dist).clamp(1e-5, 1e5))

            return w, slope

    def sample_emission(self, full_design, w, slope):
        design, slope_design = full_design[..., :self.div], full_design[..., self.div:]
        batch_shape = design.shape[:-2]
        obs_sd = pyro.param(self.prefix+"obs_sd").expand(batch_shape).unsqueeze(-1)

        ###################
        # Sigmoid transform
        ###################
        # Run the regressor forward conditioned on inputs
        prediction_mean = rmv(design, w)
        response_dist = dist.CensoredSigmoidNormal(
            loc=slope*prediction_mean, scale=slope*obs_sd, upper_lim=1.-EPSILON, lower_lim=EPSILON
        ).to_event(1)
        return pyro.sample("y", response_dist)

    def model(self, design):

        with poutine.block(**self.block_kwargs):
            w, slope = self.sample_latents(design)
            return self.sample_emission(design, w, slope)


class OldParticipantModel(NewParticipantModel):

    def __init__(self, prefix, div, **block_kwargs):
        super(OldParticipantModel, self).__init__(prefix, div, **block_kwargs)

    def sample_latents(self, full_design):
        design, slope_design = full_design[..., :self.div], full_design[..., self.div:]
        # design is size batch x n x p
        n, p = design.shape[-2:]
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            # Build the regression coefficient
            w = []

            ###############
            # Fixed effects
            ###############
            fixed_effect_mean = pyro.param(self.prefix+"fixed_effect_mean")
            fixed_effect_dist = dist.MultivariateNormal(
                fixed_effect_mean.expand(batch_shape + (fixed_effect_mean.shape[-1],)),
                scale_tril=rtril(pyro.param(self.prefix+"fixed_effect_scale_tril")))
            w.append(pyro.sample("fixed_effects", fixed_effect_dist))

            ################
            # Random effects
            ################
            re_precision_dist = dist.Gamma(
                pyro.param(self.prefix+"random_effect_precision_alpha").expand(batch_shape),
                pyro.param(self.prefix+"random_effect_precision_beta").expand(batch_shape))
            pyro.sample("random_effects_precision", re_precision_dist)
            # Sample this random effect from a MV Normal, conditional on fixed effects
            delta = w[0] - fixed_effect_mean
            mixing_matrix = pyro.param(self.prefix+"mixing_matrix")
            random_effect_mean = pyro.param(self.prefix+"random_effect_mean") + rmv(mixing_matrix, delta)
            # Masking matrix makes the posterior independent for different participants
            masking = rtril(masking_matrix(fixed_effect_mean.shape[-1], random_effect_mean.shape[-1]))
            re_dist = dist.MultivariateNormal(
                random_effect_mean.expand(batch_shape + (random_effect_mean.shape[-1],)),
                scale_tril=masking*pyro.param(self.prefix+"random_effect_scale_tril"))
            w.append(pyro.sample("random_effects", re_dist))

            # Regression coefficient `w` is batch x p
            w = broadcast_cat(w)

            ##############
            # Random slope
            ##############
            slope_precision_dist = dist.Gamma(
                pyro.param(self.prefix+"slope_precision_alpha").expand(batch_shape),
                pyro.param(self.prefix+"slope_precision_beta").expand(batch_shape))
            pyro.sample("random_slope_precision", slope_precision_dist)
            # Sample random slope from its own, independent distribution
            target_shape = batch_shape + (slope_design.shape[-1],)
            slope_dist = dist.LogNormal(pyro.param(self.prefix+"slope_mean").expand(target_shape),
                                        pyro.param(self.prefix+"slope_sd").expand(target_shape)).to_event(1)
            slope = rmv(slope_design, pyro.sample("random_slope", slope_dist).clamp(1e-5, 1e5))

            return w, slope


# TODO make this match the sprites we have
# A lot of the details are wrapped into the design matrix
# The following function builds design matrices from a simpler description
# of "left" and "right" properties.
FEATURES = [("brows", ["thin_black", "surprise_black", "thick_black"]),
            ("mouth", ["smile", "frown", "upper_teeth"])]
FIXED = [("eyes", "thin_black"), ("hair", "spiky_brown"), ("nose", "medium_brown"), ("shirt", "red")]
FLATTENED = [(group, feature) for (group, grouplist) in FEATURES for feature in grouplist]
INDEX = {x: i for i, x in enumerate(FLATTENED)}


def log_check_pyro_param_store(output):
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        value = pyro.param(name)
        output[name] = value
        if torch.isnan(value).any() or (value == float('inf')).any() or (value == float('-inf')).any():
            raise ArithmeticError("Found invalid param value {} {}".format(name, value))


def gen_design_space():
    secs, opts = list(zip(*FEATURES))
    one_side = itertools.product(*opts)
    one_side_f = [{sec: opt for sec, opt in zip(secs, optp)} for optp in one_side]
    pairs = itertools.combinations(one_side_f, 2)
    final = [[list(x)] for x in pairs]
    return final


def design_matrix(design_spec, participant_number, total_num_participants):
    n = len(design_spec)
    p = len(FLATTENED)
    p_tot = p * (1 + total_num_participants)
    matrix = torch.zeros(n, p)
    for i, spec in enumerate(design_spec):
        # Left
        for group, feature in spec[0].items():
            j = INDEX[(group, feature)]
            matrix[i, j] += 1.
        # Right
        for group, feature in spec[1].items():
            j = INDEX[(group, feature)]
            matrix[i, j] -= 1.
    # Random effects, add a copy of the whole matrix
    output = torch.zeros(n, p_tot)
    output[:, 0:p] = matrix
    output[:, ((participant_number - 1)*p):(participant_number*p)] = matrix
    participant_matrix = torch.zeros(n, total_num_participants)
    participant_matrix[..., participant_number-1] = 1.
    output = torch.cat([output, participant_matrix], dim=-1)
    return output


def masking_matrix(p, P):
    M = torch.zeros(P, P)
    for i in range(0, P, p):
        M[i:(i+p),i:(i+p)] = 1.
    return M


def design_png(design_spec):
    order = ["eyes", "brows", "hair", "nose", "mouth", "shirt"]
    dct = dict(FIXED)
    dct.update(design_spec)
    name = "sprite."+".".join(["_".join([feature, dct[feature]]) for feature in order])+".png"
    return name


def true_model(p, p_re, num_participants):

    # Re is batch x p_re
    re_sd = rexpand(pyro.param("true_re_sigma"), p_re)
    re_dist = dist.Normal(torch.zeros(p_re), re_sd).to_event(1)
    re = pyro.sample("true_random_effects", re_dist)

    ##############
    # Random slope
    ##############
    # Slope is batch x num_participants
    slope_sd_expanded = rexpand(pyro.param("true_slope_sd"), num_participants)
    slope_dist = dist.LogNormal(0., slope_sd_expanded).to_event(1)
    slope = pyro.sample("true_random_slope", slope_dist)

    def inner_true_model(full_design):
        design, slope_design = full_design[..., :(p+p_re)], full_design[..., (p+p_re):]
        # design is size batch x n x p
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            # response will be shape batch x n
            obs_sd = pyro.param("true_obs_sd").expand(batch_shape).unsqueeze(-1)

            # Build the regression coefficient
            true_fixed_effects = pyro.param("true_fixed_effects")
            w = [true_fixed_effects.expand(batch_shape + (true_fixed_effects.shape[-1],))]
            w.append(re.expand(batch_shape + (re.shape[-1],)))
            w = broadcast_cat(w)

            ###################
            # Sigmoid transform
            ###################
            # Run the regressor forward conditioned on inputs
            prediction_mean = rmv(design, w)
            this_slope = rmv(slope_design, slope)
            response_dist = dist.CensoredSigmoidNormal(
                loc=this_slope*prediction_mean, scale=this_slope*obs_sd, upper_lim=1.-EPSILON, lower_lim=EPSILON
            ).to_event(1)
            return pyro.sample("y", response_dist)

    return inner_true_model


def main(num_runs, num_parallel, num_participants, num_questions, experiment_name):
    output_dir = "./run_outputs/turk_simulation/"
    if not experiment_name:
        experiment_name = output_dir+"{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir+experiment_name
    logging.basicConfig(filename=experiment_name+'.log', level=logging.DEBUG)
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    results_file = experiment_name+'.result_stream.pickle'

    print("Experiment", experiment_name)
    #typs = ['hist'+SRC]
    typs = ['oed_no_re', 'oed', 'rand']
    logging.info("Types: {}, num runs: {}, num_parallel: {}, "
                 "num participants: {}, num questions: {}".format(
                    typs, num_runs, num_parallel, num_participants, num_questions
    ))

    CANDIDATE_DESIGNS = gen_design_space()
    N_DESIGNS = len(CANDIDATE_DESIGNS)
    N_FEATURES = len(FLATTENED)
    ALL_LATENTS = ["fixed_effects", "random_effects", "random_effects_precision",
                   "random_slope", "random_slope_precision"]
    p = len(FLATTENED)
    p_re = p * num_participants

    true_effects = torch.tensor([-10., 10., 0., -4., -2., 6.])
    logging.info("True fixed effects: {}".format(true_effects))
    print("True fixed effects:", true_effects)

    hist_file = open(SRC, 'rb')
    for typ in typs:
        for run_id in range(1, num_runs+1):
            # Initialise pyro parameters for model
            # All tensor-valued parameters should be pyro.param
            pyro.clear_param_store()
            # Fixed parameters
            pyro.param("true_re_sigma", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("true_slope_sd", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("true_obs_sd", lexpand(torch.tensor(10.), num_parallel, 1))
            pyro.param("true_fixed_effects", lexpand(true_effects, num_parallel, 1))
            pyro.param("prior_fixed_effect_mean", lexpand(torch.zeros(p), num_parallel, 1))
            pyro.param("prior_fixed_effect_scale_tril", 10. * lexpand(torch.eye(p), num_parallel, 1))
            pyro.param("prior_random_effect_mean", lexpand(torch.zeros(p_re), num_parallel, 1))
            pyro.param("prior_random_effect_precision_alpha", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("prior_random_effect_precision_beta", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("prior_slope_precision_alpha", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("prior_slope_precision_beta", lexpand(torch.tensor(2.), num_parallel, 1))
            pyro.param("prior_obs_sd", lexpand(torch.tensor(10.), num_parallel, 1))
            # Variables
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
            pyro.param("model_slope_sd", lexpand(4.*torch.ones(num_participants), num_parallel, 1))
            pyro.param("model_mixing_matrix", lexpand(torch.zeros(p_re, p), num_parallel, 1))
            pyro.param("guide_fixed_effect_mean", lexpand(torch.zeros(p), num_parallel, 1))
            pyro.param("guide_fixed_effect_scale_tril", 10. * lexpand(torch.eye(p), num_parallel, 1))
            pyro.param("guide_random_effect_mean", lexpand(torch.zeros(p_re), num_parallel, 1))
            pyro.param("guide_random_effect_precision_alpha", lexpand(torch.tensor(2.), num_parallel, 1),
                       constraint=constraints.positive)
            pyro.param("guide_random_effect_precision_beta", lexpand(torch.tensor(2.), num_parallel, 1),
                       constraint=constraints.positive)
            pyro.param("guide_slope_precision_alpha", lexpand(torch.tensor(2.), num_parallel, 1),
                       constraint=constraints.positive)
            pyro.param("guide_slope_precision_beta", lexpand(torch.tensor(2.), num_parallel, 1),
                       constraint=constraints.positive)
            pyro.param("guide_random_effect_scale_tril", lexpand(
                torch.eye(p_re, p_re), num_parallel, 1))
            pyro.param("guide_slope_mean", lexpand(torch.zeros(num_participants), num_parallel, 1))
            pyro.param("guide_slope_sd", lexpand(4.*torch.ones(num_participants), num_parallel, 1),
                       constraint=constraints.positive)
            pyro.param("guide_mixing_matrix", lexpand(torch.zeros(p_re, p), num_parallel, 1))

            # Other magic numbers
            marginal_mu_init, marginal_sigma_init = 0., 25.
            like_mu_init, like_sigma_init = 0., 10.
            rainforth_m = 100
            oed_n_samples, oed_n_steps, oed_final_n_samples, oed_lr = 10, 1500, 2000, 0.05
            oednore_n_samples, oednore_n_steps, oednore_final_n_samples, oednore_lr = 10, 750, 2000, 0.05
            elbo_n_samples, elbo_n_steps, elbo_lr = 10, 750, 0.05

            logging.info("Parameter initial values")
            for name in sorted(pyro.get_param_store().get_all_param_names()):
                value = pyro.param(name)
                logging.info("{} ({}):\n{}".format(name, value.shape, value[0, 0, ...]))

            sigmoid_response_est = SigmoidMarginalGuide(
                (num_parallel, N_DESIGNS), {"y": 1}, mu_init=marginal_mu_init, sigma_init=marginal_sigma_init
            )
            sigmoid_likelihood_est = SigmoidLikelihoodGuide(
                (num_parallel, N_DESIGNS), {"fixed_effects": N_FEATURES, "random_effects": N_FEATURES}, {"y": 1},
                mu_init=like_mu_init, sigma_init=like_sigma_init
            )
            logging.info("Marginal init values: mu_init {}, sigma_init {}".format(marginal_mu_init, marginal_sigma_init))
            logging.info("Likelihood init value: mu_init {} sigma_init {}".format(like_mu_init, like_sigma_init))

            prior = NewParticipantModel("prior_", p+p_re, hide_fn=lambda s: s["name"].startswith("prior"))
            d_star_designs = torch.tensor([])
            ys = torch.tensor([])

            # Sample re for this participant
            participant_true_model = true_model(p, p_re, num_participants)

            for participant_number in range(1, num_participants+1):
                design = torch.stack([design_matrix(d, participant_number, num_participants) for d in CANDIDATE_DESIGNS], dim=0)
                logging.info("Design shape: {}\nDesign matrix: {}".format(design.shape, design))
                adesign = lexpand(design, num_parallel)

                logging.debug("Update model to NewParticipant")
                model = NewParticipantModel("model_", p+p_re, hide_fn=lambda s: s["name"].startswith("model"))

                for question_number in range(1, num_questions+1):
                    print("Type {} run {} of {} participant {} of {} question {} of {}".format(
                        typ, run_id, num_runs, participant_number, num_participants, question_number, num_questions)
                    )
                    logging.info("Type {} run {} of {} participant {} of {} question {} of {}".format(
                        typ, run_id, num_runs, participant_number, num_participants, question_number, num_questions)
                    )
                    results = {'typ': typ, 'run': run_id, 'participant': participant_number, 'question': question_number}

                    mult = 5 if question_number == 1 else 1
                    if typ == 'oed':
                        # Throws ArithmeticError if NaN encountered
                        estimation_surface = gibbs_y_re_eig(
                            model.model, adesign, "y", "fixed_effects",
                            oed_n_samples, mult*oed_n_steps, sigmoid_response_est, sigmoid_likelihood_est,
                            optim.Adam({"lr": oed_lr}), False, None, oed_final_n_samples
                        )
                        logging.info("Running gibbs_y_re_eig with n_samples {} n_steps {} "
                                     "final_n_samples {} lr {}".format(oed_n_samples, mult*oed_n_steps,
                                                                       oed_final_n_samples, oed_lr))
                        results['estimation_surface'] = estimation_surface
                        print("EIG surface", estimation_surface)
                        if (estimation_surface < 0.).all():
                            logging.warn("All EIG values are <0 for at least one parallel run")

                        if CHECK_RAINFORTH:
                            check_surface = naive_rainforth_eig(
                                model.model, adesign, ["y"], ["fixed_effects"],
                                rainforth_m*rainforth_m, rainforth_m, rainforth_m
                            )
                            logging.info("Run Rainforth+RE with M={}".format(rainforth_m))
                            logging.info("Estimation surface from Rainforth\n{}".format(check_surface))

                        d_star_index = torch.argmax(estimation_surface, dim=1)

                    elif typ == 'oed_no_re':
                        # Throws ArithmeticError if NaN encountered
                        estimation_surface = gibbs_y_eig(
                            model.model, adesign, ["y"], ALL_LATENTS, oednore_n_samples, mult*oednore_n_steps,
                            sigmoid_response_est, optim.Adam({"lr": oednore_lr}), False, None,
                            oednore_final_n_samples
                        )
                        logging.info("Running gibbs_y_eig with n_samples {} n_steps {} "
                                     "final_n_samples {} lr {}".format(oednore_n_samples, mult*oednore_n_steps,
                                                                       oednore_final_n_samples, oednore_lr))
                        results['estimation_surface'] = estimation_surface
                        print("EIG surface", estimation_surface)
                        if (estimation_surface < 0.).all():
                            logging.warn("All EIG values are <0")

                        if CHECK_RAINFORTH:
                            check_surface = naive_rainforth_eig(
                                model.model, adesign, ["y"], ALL_LATENTS, rainforth_m*rainforth_m, rainforth_m
                            )
                            logging.info("Run Rainforth with M={}".format(rainforth_m))
                            logging.info("Estimation surface from Rainforth\n{}".format(check_surface))

                        d_star_index = torch.argmax(estimation_surface, dim=1)

                    elif typ == 'rand':
                        d_star_index = torch.randint(N_DESIGNS, (num_parallel, )).long()

                    elif typ.startswith('hist'):
                        hist = pickle.load(hist_file)
                        d_star_index = hist['d_star_index']

                    logging.info("Select index {}".format(d_star_index))
                    results['d_star_index'] = d_star_index
                    d_star_design = design[d_star_index, ...].unsqueeze(1)
                    results['d_star_design'] = d_star_design
                    d_star_designs = torch.cat([d_star_designs, d_star_design], dim=-2)
                    if typ.startswith('hist'):
                        y = hist['y']
                    else:
                        y = participant_true_model(d_star_design)
                    ys = torch.cat([ys, y], dim=-1)
                    results['y'] = y

                    # Do inference
                    guide = OldParticipantModel("guide_", p+p_re, hide=[])

                    # Not using pyro.infer.SVI: vectorization not as convenient as with
                    # own method, which I'm using for now

                    # Throws ArithmeticError - handle outside
                    elbo_learn(
                        prior.model, d_star_designs, ["y"], ALL_LATENTS, elbo_n_samples, elbo_n_steps,
                        guide.sample_latents, {"y": ys}, optim.Adam({"lr": elbo_lr})
                    )
                    logging.info("Ran elbo_learn with n_samples {} n_steps {} lr {}".format(
                        elbo_n_samples, elbo_n_steps, elbo_lr
                    ))
                    # Throws ArithmeticError
                    log_check_pyro_param_store(results)
                    print("Fixed effect mean", pyro.param("guide_fixed_effect_mean").squeeze())
                    print("Slope hyperparameters", pyro.param("guide_slope_precision_alpha"),
                                                   pyro.param("guide_slope_precision_beta"))

                    # Set the model to be the guide with fixed parameter values
                    param_store = pyro.get_param_store()
                    suffices = [name.split("_", 1)[1] for name in param_store.get_all_param_names() \
                                if name.startswith("guide")]
                    for suffix in suffices:
                        param_store._params["model_"+suffix].data[...] = pyro.param("guide_"+suffix).data[...]

                    logging.debug("Update model to OldParticipant")
                    model = OldParticipantModel("model_", p+p_re, hide_fn=lambda s: s["name"].startswith("model"))

                    logging.debug("Store results to pickle stream")
                    with open(results_file, 'ab') as f:
                        pickle.dump(results, f)
    hist_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design")
    parser.add_argument("--num-runs", nargs="?", default=1, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=5, type=int)
    parser.add_argument("--num-participants", nargs="?", default=10, type=int)
    parser.add_argument("--num-questions", nargs="?", default=5, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    args = parser.parse_args()
    main(args.num_runs, args.num_parallel, args.num_participants, args.num_questions, args.name)
