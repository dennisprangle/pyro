from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

output_dir = "./run_outputs/location/"
COLOURS = [[31/255, 120/255, 180/255], [227/255, 26/255, 28/255], [51/255, 160/255, 44/255], [177/255, 89/255, 40/255],
           [106 / 255, 61 / 255, 154 / 255], [255/255, 127/255, 0], [.22, .22, .22]]
VALUE_LABELS = {"Entropy": "Posterior entropy on fixed effects",
                "L2 distance": "Expected L2 distance from posterior to truth",
                "Optimized EIG": "Maximized EIG",
                "EIG gap": "Difference between maximum and mean EIG",
                "Fixed effects @0": "Fixed effects index 1",
                "Fixed effects @3": "Fixed effects index 4"}
LABELS = {'oed': 'OED', 'posterior_mean': 'Posterior mean', 'rand': 'Random'}
MARKERS = ['o', 'D', '^']

S = 3


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


def main(fnames, findices, plot):
    fnames = fnames.split(", ")
    findices = map(int, findices.split(", "))

    if not all(fnames):
        results_fnames = sorted(glob.glob(output_dir+"*.result_stream.pickle"))
        fnames = [results_fnames[i] for i in findices]
    else:
        fnames = [output_dir+name+".result_stream.pickle" for name in fnames]

    if not fnames:
        raise ValueError("No matching files found")

    results_dict = defaultdict(list)
    for fname in fnames:
        with open(fname, 'rb') as results_file:
            try:
                while True:
                    results = pickle.load(results_file)
                    # Compute entropy and L2 distance to the true fixed effects
                    mean = results["mean"]
                    sd = results["sd"]
                    entropy = .5*torch.log(2*np.pi*np.e*sd**2)
                    design = results['d_star_design'].squeeze()
                    output = {"Entropy": entropy, "mean": mean, "sd": sd, "d": design}
                    results_dict[results['typ']].append(output)
            except EOFError:
                continue

    # Get results into better format
    # First, concat across runs
    possible_stats = list(set().union(a for v in results_dict.values() for a in v[0].keys()))
    reformed = {statistic: {
        k: torch.stack([a[statistic] for a in v]).detach().numpy()
        for k, v in results_dict.items() if statistic in v[0]}
        for statistic in possible_stats}

    if plot:
        # Plot designs and posteriors
        for k in reformed["mean"]:
            plt.figure(figsize=(8, 5))
            m = reformed["mean"][k][:, 0, 0, 0]
            s = reformed["sd"][k][:, 0, 0, 0]
            d = reformed["d"][k][:, 0]

            x = np.arange(0, m.shape[0])
            plt.plot(x, m, linestyle='-', markersize=8, color=COLOURS[1], marker='x', linewidth=2, mew=2)
            plt.fill_between(x, m-1.65*s, m+1.65*s, color=COLOURS[1]+[.2])
            plt.plot(x, d, linestyle='', markersize=8, color=COLOURS[0], marker='x', linewidth=2, mew=2)
            # plt.title(value_label, fontsize=18)
            plt.xlabel("Step", fontsize=22)
            plt.ylabel("Design $d_t$", fontsize=22)
            plt.xticks(fontsize=16)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.yticks(fontsize=16)
            # [i.set_linewidth(S/2) for i in plt.gca().spines.values()]
            # plt.gca().tick_params(width=S/2)

            plt.show()
        plt.figure(figsize=(8, 5))
        for i, k in enumerate(reformed["Entropy"]):
            e = reformed["Entropy"][k].squeeze()
            centre = e.mean(1)
            upper, lower = np.percentile(e, 95, axis=1), np.percentile(e, 5, axis=1)
            x = np.arange(0, e.shape[0])
            plt.plot(x, centre, linestyle='-', markersize=8, color=COLOURS[i], marker=MARKERS[i], linewidth=2)
            plt.fill_between(x, upper, lower, color=COLOURS[i] + [.2])
        plt.xlabel("Step", fontsize=22)
        plt.xticks(fontsize=16)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend([LABELS[k] for k in reformed["Entropy"].keys()], fontsize=16, frameon=False)
        # frame = legend.get_frame()
        # frame.set_linewidth(S/)
        plt.yticks(fontsize=16)
        plt.ylabel("Posterior entropy", fontsize=22)
        # [i.set_linewidth(S/2) for i in plt.gca().spines.values()]
        # plt.gca().tick_params(width=S/2)
        plt.show()


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
