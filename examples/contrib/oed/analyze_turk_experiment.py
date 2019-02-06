from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pyro
import pyro.distributions as dist
from pyro.contrib.util import rmv, rmm, rtril, iter_iaranges_to_shape, rexpand
from pyro.ops.linalg import rinverse
from pyro.contrib.oed.eig import elbo_learn

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2

output_dir = "./run_outputs/turk_simulation/"
COLOURS = [[1, .6, 0], [1, .4, .4], [.5, .5, 1.], [1., .5, .5]]
VALUE_LABELS = {"Entropy": "Posterior entropy on fixed effects",
                "L2 distance": "Expected L2 distance from posterior to truth",
                "Optimized EIG": "Maximized EIG",
                "EIG gap": "Difference between maximum and mean EIG"}
EPSILON = torch.tensor(2 ** -24)


def upper_lower(array):
    centre = array.mean(1)
    upper, lower = np.percentile(array, 95, axis=1), np.percentile(array, 5, axis=1)
    return lower, centre, upper


def rlogdet(M):
    old_shape = M.shape[:-2]
    tbound = M.view(-1, M.shape[-2], M.shape[-1])
    ubound = torch.unbind(tbound, dim=0)
    logdets = [torch.logdet(m) for m in ubound]
    bound = torch.stack(logdets)
    return bound.view(old_shape)


def rtrace(M):
    old_shape = M.shape[:-2]
    tbound = M.view(-1, M.shape[-2], M.shape[-1])
    ubound = torch.unbind(tbound, dim=0)
    traces = [torch.trace(m) for m in ubound]
    bound = torch.stack(traces)
    return bound.view(old_shape)


div = 6


def model(full_design):

    X, slope_design = full_design[..., :div], full_design[..., div:]
    batch_shape = X.shape[:-2]
    with ExitStack() as stack:
        for iarange in iter_iaranges_to_shape(batch_shape):
            stack.enter_context(iarange)

        fixed_effect_mean = torch.tensor([0.,0.,0.,0.,0.,0.])
        fixed_effect_scale_tril = 10.*torch.eye(6)
        fixed_effect_dist = dist.MultivariateNormal(
                    fixed_effect_mean.expand(batch_shape + (fixed_effect_mean.shape[-1],)),
                    scale_tril=rtril(fixed_effect_scale_tril))
        w = pyro.sample("fixed_effects", fixed_effect_dist)

        ##############
        # Random slope
        ##############
        slope_precision_dist = dist.Gamma(
            torch.tensor(1.).expand(batch_shape),
            torch.tensor(1.).expand(batch_shape))
        slope_precision = pyro.sample("random_slope_precision", slope_precision_dist)
        slope_sd = rexpand(1./torch.sqrt(slope_precision), slope_design.shape[-1])
        slope_dist = dist.LogNormal(-.5*slope_sd.pow(2), slope_sd).independent(1)
        slope = rmv(slope_design, pyro.sample("random_slope", slope_dist))

        obs_sd = torch.tensor(10.)
        prediction_mean = rmv(X, w)
        response_dist = dist.CensoredSigmoidNormal(
            loc=slope*prediction_mean, scale=slope*obs_sd,
            upper_lim=1.-EPSILON, lower_lim=EPSILON).independent(1)
        return pyro.sample("y", response_dist)


def guide(full_design):

    X, slope_design = full_design[..., :div], full_design[..., div:]
    batch_shape = X.shape[:-2]
    with ExitStack() as stack:
        for iarange in iter_iaranges_to_shape(batch_shape):
            stack.enter_context(iarange)

        mean = pyro.param("fixed_effect_mean", torch.tensor([0.,0.,0.,0.,0.,0.]))
        scale_tril = pyro.param("fixed_effect_scale_tril", 10.*torch.eye(6))
        pyro.sample("fixed_effects", dist.MultivariateNormal(
            mean.expand(batch_shape + (mean.shape[-1],)), scale_tril=rtril(scale_tril)))

        ##############
        # Random slope
        ##############
        slope_prec_alpha = pyro.param("slope_precision_alpha", torch.tensor(1.)).expand(batch_shape)
        slope_prec_beta = pyro.param("slope_precision_beta", torch.tensor(1.)).expand(batch_shape)
        slope_precision_dist = dist.Gamma(slope_prec_alpha, slope_prec_beta)
        pyro.sample("random_slope_precision", slope_precision_dist)
        # Sample random slope from its own, independent distribution
        target_shape = batch_shape + (slope_design.shape[-1],)
        slope_mean = pyro.param("slope_mean", torch.zeros(slope_design.shape[-1]))
        slope_sd = pyro.param("slope_sd", 4.*torch.ones(slope_design.shape[-1]))

        slope_dist = dist.LogNormal(slope_mean.expand(target_shape),
                                    slope_sd.expand(target_shape)).independent(1)
        pyro.sample("random_slope", slope_dist)


def log_check_pyro_param_store(output):
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        value = pyro.param(name)
        output[name] = value.clone()
        if torch.isnan(value).any() or (value == float('inf')).any() or (value == float('-inf')).any():
            raise ArithmeticError("Found invalid param value {} {}".format(name, value))


def main(fnames, findices, plot):

    make_mean = torch.cat([torch.cat([(1./3)*torch.ones(3, 3), torch.zeros(3, 3)], dim=0),
                           torch.cat([torch.zeros(3, 3), (1./3)*torch.ones(3, 3)], dim=0)], dim=1)

    fnames = fnames.split(",")
    findices = map(int, findices.split(","))

    if not all(fnames):
        results_fnames = sorted(glob.glob(output_dir+"*.result_stream.pickle"))
        fnames = [results_fnames[i] for i in findices]
    else:
        fnames = [output_dir+name+".result_stream.pickle" for name in fnames]

    if not fnames:
        raise ValueError("No matching files found")

    X = torch.tensor([])
    y = torch.tensor([])
    for fname in fnames:
        with open(fname, 'rb') as results_file:
            try:
                while True:
                    results = pickle.load(results_file)
                    if results['typ'] == 'rand':
                        X = torch.cat([X, results['d_star_design']], dim=0)
                        y = torch.cat([y, results['y']], dim=0)
            except EOFError:
                continue

    hist = defaultdict(dict)
    for i in [1,2,3,4,5, 10, 15, 20, 30, 40, 42, 45, 50, 52, 55, 60, 62, 65, 70, 75, 80]:
        X, y = X.squeeze(), y.squeeze()
        X_fix = X[0:i, 0:6]
        X_fix2 = torch.cat([X[0:i, 0:6], X[0:i, -8:]], dim=1)
        y_fix = y[0:i]
        yt = y_fix.log() - (1. - y_fix).log()

        print(X.shape, y_fix.shape, X_fix.shape)
        beta_hat = rmv(rinverse(rmm(X_fix.transpose(0,1), X_fix) + 0.01*torch.eye(6)), rmv(X_fix.transpose(0,1), yt))
        hist[i]["beta hat"] = beta_hat
        print('fixed effects ridge', beta_hat)

        elbo_learn(model, X_fix2, ["y"], ["fixed_effects", "random_slope_precision", "random_slope"], 10, 600,
                   guide, {"y": y_fix}, pyro.optim.Adam({"lr": 0.05}))

        log_check_pyro_param_store(hist[i])
        st = pyro.param("fixed_effect_scale_tril")
        covm = torch.matmul(st, st.transpose(-1, -2))
        hist[i]['entropy'] = .5 * rlogdet(2 * np.pi * np.e * covm).squeeze(-1)

    for k, v in hist.items():
        print(k)
        for j, w in v.items():
            print(j, w)

    if plot:
        pass
        # for k, r in descript.items():
        #     value_label = VALUE_LABELS[k]
        #     plt.figure(figsize=(10, 5))
        #     for i, (lower, centre, upper) in enumerate(r.values()):
        #         x = np.arange(0, centre.shape[0])
        #         plt.plot(x, centre, linestyle='-', markersize=6, color=COLOURS[i], marker='o')
        #         plt.fill_between(x, upper, lower, color=COLOURS[i]+[.2])
        #     # plt.title(value_label, fontsize=18)
        #     plt.legend(r.keys(), loc=1, fontsize=16)
        #     plt.xlabel("Step", fontsize=18)
        #     plt.ylabel(value_label, fontsize=18)
        #     plt.xticks(fontsize=14)
        #     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        #     plt.yticks(fontsize=14)
        #     plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design results parser")
    parser.add_argument("--fnames", nargs="?", default="", type=str)
    parser.add_argument("--findices", nargs="?", default="-1", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.fnames, args.findices, args.plot)
