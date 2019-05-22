from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

output_dir = "./run_outputs/"
lightblue = np.array([158, 202, 225])/255
darkblue = np.array([0, 0, 0])/255
COLOURS = {
           0: lightblue,
           250: .5*lightblue + .5*darkblue,
           150: [51/255, 160/255, 44/255],
           125: .75*lightblue + .25*darkblue,
           100: [(44+197)/510, (127+27)/510, (184+138)/510],
           500: .25*lightblue + .75*darkblue,
           2500: darkblue,
           "Ground truth": [0., 0., 0.],
           "Nested Monte Carlo": [227/255, 26/255, 28/255],
           "posterior": [31/255, 120/255, 180/255],
           "Posterior exact guide": [1, .4, .4],
           "marginal": [51/255, 160/255, 44/255],
           "Marginal + likelihood": [.1, .7, .4],
           "Amortized LFIRE": [.66, .82, .43],
           "ALFIRE 2": [.3, .7, .9],
           "LFIRE": [177/255, 89/255, 40/255],
           "LFIRE 2": [.78, .40, .8],
           "IWAE": [106/255, 61/255, 154/255],
           "Laplace": [255/255, 127/255, 0],
}
MARKERS = {
           "Ground truth": 'x',
           "Nested Monte Carlo": 'v',
           "posterior": 'o',
           "Posterior exact guide": 'x',
           "marginal": 's',
           "Marginal + likelihood": 's',
           "Amortized LFIRE": 'D',
           "ALFIRE 2": 'D',
           "LFIRE": 'D',
           "LFIRE 2": 'D',
           "IWAE": '+',
           "Laplace": '*',
}


def upper_lower(array):
    centre = np.sqrt((array**2).mean(0))
    se = array.std(0)/np.sqrt(array.shape[0])
    upper, lower = centre + se, centre - se
    return lower, centre, upper


def bias_variance(array):
    mean = array.mean(0).mean(0)
    var = (array.std(0)**2).mean(0)
    return mean, var


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
                    num_steps = results['num_steps']
                    t = results['elapsed']
                    surface = results['surface']
                    results_dict[num_steps][t] = surface
            except EOFError:
                continue

    # Get results into better format
    # First, concat across runs
    print(results_dict)
    reformed = OrderedDict([(num_steps, (
        np.array([k for k in sorted(d)]),
        upper_lower(np.abs(-4.526732444763 +
                           torch.stack([v[1] for v in sorted(d.items())], dim=-1).squeeze().detach().numpy()))))
        for num_steps, d in sorted(results_dict.items())])
    print(reformed)

    if plot:
        plt.figure(figsize=(5, 3))
        for num_steps, (x, (lower, centre, upper)) in reformed.items():
            plt.plot(x, centre, linestyle='-', markersize=8, color=COLOURS[num_steps], marker='x', linewidth=2, mew=2)
            plt.fill_between(x, upper, lower, color=list(COLOURS[num_steps])+[.15])
        plt.legend([str(x).title() + ' steps' for x in sorted(reformed.keys())], loc=3, fontsize=14, frameon=False)
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("RMSE in EIG estimate", fontsize=20)
        plt.xticks(fontsize=16)
        # plt.axhline(4.5267, color="k", linestyle='--')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.yticks(fontsize=16)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIG estimation benchmarking experiment design results parser")
    parser.add_argument("--fnames", nargs="?", default="", type=str)
    parser.add_argument("--findices", nargs="?", default="-1", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    args = parser.parse_args()
    main(args.fnames, args.findices, args.plot)
