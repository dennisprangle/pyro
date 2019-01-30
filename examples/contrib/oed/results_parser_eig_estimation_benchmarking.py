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

output_dir = "./run_outputs/eig_benchmark/"
COLOURS = {
           "Ground truth": [0., 0., 0.],
           "Nested Monte Carlo": [1, .6, 0],
           "Posterior": [1, .5, .4],
           "Marginal": [.5, .5, 1.],
           "Marginal + likelihood": [.1, .7, .4],
           "Amortized LFIRE": [.66, .82, .43],
           "ALFIRE 2": [.3, .7, .9],
           "LFIRE": [.78, .78, .60],
           "LFIRE 2": [.78, .40, .8],
           "IWAE": [.7, .4, 1.],
           "Laplace": [.9, 0., .3],
}
MARKERS = {
           "Ground truth": 'x',
           "Nested Monte Carlo": 'v',
           "Posterior": 'o',
           "Marginal": 's',
           "Marginal + likelihood": 's',
           "Amortized LFIRE": 'D',
           "ALFIRE 2": 'D',
           "LFIRE": 'D',
           "LFIRE 2": 'D',
           "IWAE": '+',
           "Laplace": '*',
}


def upper_lower(array):
    centre = array.mean(0)
    upper, lower = np.percentile(array, 95, axis=0), np.percentile(array, 5, axis=0)
    return lower, centre, upper


def bias_variance(array):
    mean = array.mean(0).mean(0)
    var = (array.std(0)**2).mean(0)
    return mean, var


def main(fnames, findices, plot):
    fnames = fnames.split(",")
    findices = map(int, findices.split(","))

    if not all(fnames):
        results_fnames = sorted(glob.glob(output_dir+"*.result_stream.pickle"))
        fnames = [results_fnames[i] for i in findices]
    else:
        fnames = [output_dir+name+".result_stream.pickle" for name in fnames]

    if not fnames:
        raise ValueError("No matching files found")

    results_dict = defaultdict(lambda: defaultdict(dict))
    designs = {}
    for fname in fnames:
        with open(fname, 'rb') as results_file:
            try:
                while True:
                    results = pickle.load(results_file)
                    case = results['case']
                    estimator = results['estimator_name']
                    run_num = results['run_num']
                    results_dict[case][estimator][run_num] = results['surface']
                    designs[case] = results['design']
            except EOFError:
                continue

    # Get results into better format
    # First, concat across runs
    reformed = {case: OrderedDict([
                    (estimator, upper_lower(torch.cat([v[run] for run in v]).detach().numpy()))
                    for estimator, v in sorted(d.items())])
                for case, d in results_dict.items()
                }

    if plot:
        for case, d in reformed.items():
            plt.figure(figsize=(10, 5))
            for k, (lower, centre, upper) in d.items():
                # x = designs[case][:,0,0].numpy()
                x = np.arange(0, centre.shape[0])
                plt.plot(x, centre, linestyle='-', markersize=6, color=COLOURS[k], marker=MARKERS[k])
                plt.fill_between(x, upper, lower, color=COLOURS[k]+[.2])
            plt.title(case, fontsize=18)
            plt.legend(d.keys(), loc=2, fontsize=14)
            plt.xlabel("Design $d$", fontsize=18)
            plt.ylabel("EIG estimate", fontsize=18)
            plt.xticks(fontsize=14)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.yticks(fontsize=14)
            plt.show()
    else:
        print(reformed)
        # truth = {case: torch.cat([d["Ground truth"][run] for run in d["Ground truth"]]) for case, d in results_dict.items()}
        # bias_var = {case: OrderedDict([
        #                 (estimator, bias_variance((torch.cat([v[run] for run in v]) - truth[case]).detach().numpy()))
        #                 for estimator, v in sorted(d.items())])
        #             for case, d in results_dict.items()
        #             }
        # print(bias_var)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIG estimation benchmarking experiment design results parser")
    parser.add_argument("--fnames", nargs="?", default="", type=str)
    parser.add_argument("--findices", nargs="?", default="-1", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.fnames, args.findices, args.plot)