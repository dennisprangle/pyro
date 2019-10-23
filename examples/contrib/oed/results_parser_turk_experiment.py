from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pyro.contrib.util import rmv

output_dir = "./run_outputs/turk_simulation/"
COLOURS = [[227/255, 26/255, 28/255], [31/255, 120/255, 180/255], [51/255, 160/255, 44/255], [177/255, 89/255, 40/255],
           [106 / 255, 61 / 255, 154 / 255], [255/255, 127/255, 0], [.22, .22, .22], [.44, .44, .44], [.66, .66, .66]]
COLOURSD = {'rand': [227/255, 26/255, 28/255], 'oed': [31/255, 120/255, 180/255],
            'oed_no_re': [51/255, 160/255, 44/255]}
VALUE_LABELS = {"Entropy": "Posterior entropy on fixed effects",
                "L2 distance": "Expected L2 distance from posterior to truth",
                "Optimized EIG": "Maximized EIG",
                "EIG gap": "Difference between maximum and mean EIG",
                "Fixed effects @0": "Fixed effects index 1",
                "Fixed effects @3": "Fixed effects index 4"}
LEGENDS = {'oed': 'BOED marginal + likelihood (ours)', 'rand': 'Random design (baseline)', 'oed_no_re': 'OED on all parameters'}
MARKERS = {'oed': 'o', 'rand': 'D', 'oed_no_re': '^'}


def upper_lower(array):
    centre = array.mean(1)
    se = array.std(1) / np.sqrt(array.shape[1])
    upper, lower = centre + se, centre - se
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

    results_dict = defaultdict(lambda: defaultdict(list))
    for fname in fnames:
        with open(fname, 'rb') as results_file:
            i = 0
            try:
                while True:
                    results = pickle.load(results_file)
                    # Compute entropy and L2 distance to the true fixed effects
                    st = results["model_fixed_effect_scale_tril"]
                    covm = torch.matmul(st, st.transpose(-1, -2))
                    entropy = .5*rlogdet(2*np.pi*np.e*covm).squeeze(-1)
                    centered_fixed_effects = results["model_fixed_effect_mean"] - \
                        rmv(make_mean, results["model_fixed_effect_mean"])
                    # trace = rtrace(covm)
                    output = {"Entropy": entropy}
                    # if "true_fixed_effects" in results:
                    #     l2d = torch.norm(centered_fixed_effects - results["true_fixed_effects"], dim=-1)
                    #     el2d = torch.sqrt(l2d**2 + trace)
                    #     output["L2 distance"] = el2d.squeeze(-1)
                    output['Fixed effects @0'] = centered_fixed_effects[..., 0, 0]
                    output['Fixed effects @3'] = centered_fixed_effects[..., 0, 3]
                    if 'estimation_surface' in results and results['estimation_surface'] is not None:
                        eig_star, _ = torch.max(results['estimation_surface'], dim=1)
                        eig_star = eig_star
                        eig_mean = results['estimation_surface'].mean(1)
                        output['Optimized EIG'] = eig_star
                        output["EIG gap"] = eig_star - eig_mean
                    # TODO deal with incorrect order of
                    if results['typ'].startswith('hist'):
                        results['typ'] = 'rand'
                    if len(results_dict[results['typ']][results['run']]) >= i+1:
                        d = results_dict[results['typ']][results['run']][i]
                        for k in list(d.keys()):
                            d[k] = torch.cat([d[k], output[k]])
                    else:
                        results_dict[results['typ']][results['run']].append(output)
                    i += 1
            except EOFError:
                continue

    # if 'oed_no_re' in results_dict:
    #     del results_dict['oed_no_re']
    # Get results into better format
    # First, concat across runs
    possible_stats = list(set().union(a for v in results_dict.values() for a in v[1][0].keys()))
    reformed = {statistic: {
                    k: torch.stack([torch.cat([v[run][i][statistic] for run in v]) for i in range(len(v[1]))])
                    for k, v in results_dict.items() if statistic in v[1][0]}
                for statistic in possible_stats}
    for k, v in reformed["Entropy"].items():
        print(k, v.shape)

    descript = OrderedDict([(statistic,
                             OrderedDict([(k, upper_lower(v.detach().numpy())) for k, v in sorted(sts.items())]))
                            for statistic, sts in sorted(reformed.items())])
    print(reformed['Entropy']['oed'][-1, ...])
    print(reformed['Entropy']['rand'][-1, ...])

    if plot:
        for k, r in descript.items():
            value_label = VALUE_LABELS[k]
            plt.figure(figsize=(5, 5))
            for i, (l, (lower, centre, upper)) in enumerate(r.items()):
                x = np.arange(0, centre.shape[0])
                col = COLOURSD.get(l, COLOURS[i])
                plt.plot(x, centre, linestyle='-', markersize=6, color=col, marker=MARKERS[l])
                plt.fill_between(x, upper, lower, color=col+[.2])
            # plt.title(value_label, fontsize=18)
            plt.legend([LEGENDS[k] for k in r.keys()], loc=1, fontsize=16, frameon=False)
            plt.xlabel("Step", fontsize=18)
            plt.ylabel(value_label, fontsize=18)
            # if k == "Entropy":
            #     plt.ylim(11, 22)
            plt.xticks(fontsize=14)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.yticks(fontsize=14)
            plt.show()
    else:
        print(descript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design results parser")
    parser.add_argument("--fnames", nargs="?", default="", type=str)
    parser.add_argument("--findices", nargs="?", default="-1", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    args = parser.parse_args()
    main(args.fnames, args.findices, args.plot)
