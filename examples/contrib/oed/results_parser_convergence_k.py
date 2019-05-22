from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

output_dir = "./run_outputs/"
COLOURS = {
           0: [44/255, 127/255, 184/255],
           250: [197/255, 27/255, 138/255],
           150: [51/255, 160/255, 44/255],
           125: [(44+197)/510, (127+27)/510, (184+138)/510],
           "Ground truth": [0., 0., 0.],
           "Nested Monte Carlo": [227/255, 26/255, 28/255],
           "posterior": [31/255, 120/255, 180/255],
           "Posterior exact guide": [1, .4, .4],
           "marginal": [51/255, 160/255, 44/255],
           "Marginal + likelihood": [.1, .7, .4],
           "Posterior": [31/255, 120/255, 180/255],
           "Marginal": [51/255, 160/255, 44/255],
           "Amortized LFIRE": [.66, .82, .43],
           "ALFIRE 2": [.3, .7, .9],
           "LFIRE": [177/255, 89/255, 40/255],
           "LFIRE 2": [.78, .40, .8],
           "IWAE": [106/255, 61/255, 154/255],
           "Laplace": [255/255, 127/255, 0],
}
OTHERCOLOURS = {"Posterior": [158/255, 202/255, 225/255], "Marginal": [161/255, 217/255, 155/255]}
MARKERS = ['x', 'o', '^', '*', 'v', '<', '>', 's', 'P', 'D']


def upper_lower(array):
    # array[array > 15.] = 0.
    # print(np.min(array[:, -5]), np.max(array[:, -5]))
    centre = np.sqrt((array**2).mean(0))
    se = array.std(0)/np.sqrt(array.shape[0])
    upper, lower = centre + se, centre - se
    return lower, centre, upper


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

    results_dict = defaultdict(lambda: defaultdict(dict))
    for fname in fnames:
        with open(fname, 'rb') as results_file:
            try:
                while True:
                    results = pickle.load(results_file)
                    method = results['method'].title()
                    t = results['elapsed']
                    surface = results['surface']
                    N = results['N']
                    Ni = results['Ni']
                    T = results['T']
                    Ti = results['Ti']
                    # print(N)
                    results_dict[(method, N, Ni)][t] = surface
            except EOFError:
                continue

    # Get results into better format
    # First, concat across runs
    reformed = OrderedDict([(num_steps, (
        np.array([k for k in sorted(d)]),
        upper_lower(np.abs(-4.52673244 +
                           torch.stack([v[1] for v in sorted(d.items())], dim=-1).squeeze().detach().numpy()))))
        for num_steps, d in sorted(results_dict.items())])

    if plot:
        plt.figure(figsize=(5, 5))
        for k, (x, (lower, centre, upper)) in reformed.items():
            if k[2] == 0:
                color = OTHERCOLOURS[k[0]]
            else:
                color = COLOURS[k[0]]
            plt.plot(x, centre, linestyle='-', markersize=8, color=color, marker=MARKERS[k[2]],
                     linewidth=2, mew=2)
            plt.fill_between(x, upper, lower, color=color+[.15])
        plt.legend(["{} (N={})".format(x[0], x[1]) for x in sorted(reformed.keys())], loc=3, fontsize=12, frameon=False)
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("RMSE in EIG estimate", fontsize=20)
        plt.xticks(fontsize=16)
        # plt.axhline(4.5267, color="k", linestyle='--')
        # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.yticks(fontsize=16)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence results parser")
    parser.add_argument("--fnames", nargs="?", default="", type=str)
    parser.add_argument("--findices", nargs="?", default="-1", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    args = parser.parse_args()
    main(args.fnames, args.findices, args.plot)
