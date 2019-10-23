from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

output_dir = "./run_outputs/eig_benchmark/"
COLOURS = {
           "Ground truth": [0., 0., 0.],
           "Nested Monte Carlo": [227/255, 26/255, 28/255],
           "Non-nested Monte Carlo": [227/255, 26/255, 28/255],
           "Posterior": [31/255, 120/255, 180/255],
           "Posterior exact guide": [1, .4, .4],
           "Marginal": [51/255, 160/255, 44/255],
           "Marginal (unbiased)": [51/255, 160/255, 44/255],
           "Marginal + likelihood": [.1, .7, .4],
           "Amortized LFIRE": [.66, .82, .43],
           "ALFIRE 2": [.3, .7, .9],
           "LFIRE": [177/255, 89/255, 40/255],
           "LFIRE 2": [.78, .40, .8],
           "VNMC": [106/255, 61/255, 154/255],
           "Laplace": [255/255, 127/255, 0],
           "Donsker-Varadhan": [.44, .44, .44]
}
MARKERS = {
           "Ground truth": 'x',
           "Nested Monte Carlo": 'v',
           "Non-nested Monte Carlo": 'v',
           "Posterior": 'o',
           "Posterior exact guide": 'x',
           "Marginal": 's',
           "Marginal (unbiased)": 's',
           "Marginal + likelihood": 's',
           "Amortized LFIRE": 'D',
           "ALFIRE 2": 'D',
           "LFIRE": 'D',
           "LFIRE 2": 'D',
           "VNMC": '+',
           "Laplace": '*',
           "Donsker-Varadhan": "^"
}


def upper_lower(array):
    centre = array.mean(0)
    upper, lower = np.percentile(array, 95, axis=0), np.percentile(array, 5, axis=0)
    return lower, centre, upper


def bias_variance(array):
    mean = (array.mean(0)**2).mean(0)
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
    designs = {}
    for fname in fnames:
        with open(fname, 'rb') as results_file:
            try:
                while True:
                    results = pickle.load(results_file)
                    case = results['case']
                    estimator = results['estimator_name']
                    run_num = results['run_num']
                    # results['surface'][:, 0:18] = results['surface'][:, 0:18].mean()
                    # results['surface'][:, 18:] = results['surface'][:, 18:].mean()
                    if run_num in results_dict[case][estimator]:
                        run_num = 1 + max(results_dict[case][estimator].keys())
                    results_dict[case][estimator][run_num] = results['surface']
                    # with open('./run_outputs/eig_benchmark/turktrue.result_stream.pickle', 'ab') as f:
                    #     pickle.dump(results, f)
                    # print(results['seed'])
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
            plt.figure(figsize=(8, 5))
            for k, (lower, centre, upper) in d.items():
                # x = designs[case][:, 0, 0].numpy()
                x = np.arange(0, centre.shape[0])
                plt.plot(x, centre, linestyle='-', markersize=8, color=COLOURS[k], marker=MARKERS[k], linewidth=2)
                # plt.fill_between(x, upper, lower, color=COLOURS[k]+[.15])
            # plt.title(case, fontsize=18)
            plt.legend(d.keys(), loc=1, fontsize=16, frameon=False)
            plt.xlabel("Design $d$", fontsize=22)
            plt.ylabel("EIG estimate", fontsize=22)
            plt.xticks(fontsize=16)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.yticks(fontsize=16)
            plt.show()
    else:
        # print(reformed)
        if "Ground truth" in list(results_dict.values())[0]:
            truth = {case: torch.cat([d["Ground truth"][run] for run in d["Ground truth"]]).mean(0)
                     for case, d in results_dict.items()}
        elif "Marginal (unbiased)" in list(results_dict.values())[0]:
            truth = {case: torch.cat([d["Marginal (unbiased)"][run] for run in d["Marginal (unbiased)"]]).mean(0)
                     for case, d in results_dict.items()}
        else:
            truth = defaultdict(lambda: 0)
        bias_var = {case: OrderedDict([
                        (estimator, bias_variance((torch.cat([v[run] for run in v]) - truth[case]).detach().numpy()))
                        for estimator, v in sorted(d.items())])
                    for case, d in results_dict.items()
                    }
        for case, v in bias_var.items():
            for method, bv in v.items():
                print(case, method, "bias", '{:0.2e}'.format(bv[0]), '{:0.2e}'.format(bv[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIG estimation benchmarking experiment design results parser")
    parser.add_argument("--fnames", nargs="?", default="", type=str)
    parser.add_argument("--findices", nargs="?", default="-1", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=False)
    args = parser.parse_args()
    main(args.fnames, args.findices, args.plot)
