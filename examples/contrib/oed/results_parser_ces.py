from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

output_dir = "./run_outputs/ces/"
COLOURS = [[31/255, 120/255, 180/255], [227/255, 26/255, 28/255], [51/255, 160/255, 44/255], [177/255, 89/255, 40/255],
           [106 / 255, 61 / 255, 154 / 255], [255/255, 127/255, 0], [.22, .22, .22]]
VALUE_LABELS = {"Entropy": "Posterior entropy",
                "L2 distance": "Expected L2 distance from posterior to truth",
                "Optimized EIG": "Maximized EIG",
                "EIG gap": "Difference between maximum and mean EIG",
                "rho_rmse": "RMSE in $\\rho$ estimate",
                "alpha_rmse": "RMSE in $\\mathbf{\\alpha}$ estimate",
                "slope_rmse": 'RMSE in $u$ estimate',
                "total_rmse": 'Total RMSE'}
LABELS = {'marginal': 'Marginal BO (baseline)', 'rand': 'Random design (baseline)', 'nmc': 'BOED NMC (baseline)',
          'posterior-grad': "Posterior gradient", 'nce-grad': "NCE gradient", "ace-grad": "ACE gradient"}

MARKERS = ['o', 'D', '^', '*']

S = 3


def upper_lower(array):
    centre = array.mean(1)
    se = array.std(1)/np.sqrt(array.shape[1])
    upper, lower = centre + se, centre - se
    return lower, centre, upper
    # return np.percentile(array, 75, axis=1), np.percentile(array, 50, axis=1), np.percentile(array, 25, axis=1)


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
    fnames = fnames.split(",")
    findices = map(int, findices.split(","))

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
                    if 'rho0' in results:
                        rho0, rho1, alpha_concentration = results['rho0'], results['rho1'], results['alpha_concentration']
                    else:
                        rho_concentration = results['rho_concentration']
                        rho0, rho1 = rho_concentration.unbind(-1)
                        alpha_concentration = results['alpha_concentration']
                    slope_mu, slope_sigma = results['slope_mu'], results['slope_sigma']
                    rho_dist = torch.distributions.Beta(rho0, rho1)
                    alpha_dist = torch.distributions.Dirichlet(alpha_concentration)
                    slope_dist = torch.distributions.LogNormal(slope_mu, slope_sigma)
                    rho_rmse = torch.sqrt((rho_dist.mean - torch.tensor(.9))**2 + rho_dist.variance)
                    alpha_rmse = torch.sqrt((alpha_dist.mean - torch.tensor([.2, .3, .5])).pow(2).sum(-1))
                    slope_rmse = torch.sqrt((slope_dist.mean - torch.tensor(10.)).pow(2) + slope_dist.variance)
                    total_rmse = torch.sqrt(rho_rmse**2 + alpha_rmse**2 + slope_rmse**2)
                    entropy = rho_dist.entropy() + alpha_dist.entropy() + slope_dist.entropy()
                    output = {"rho_rmse": rho_rmse, "alpha_rmse": alpha_rmse, "slope_rmse": slope_rmse,
                              "Entropy": entropy, "total_rmse": total_rmse}
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
        for statistic in ["Entropy", "rho_rmse", "alpha_rmse", "slope_rmse"]:
            plt.figure(figsize=(5, 5))
            for i, k in enumerate(reformed[statistic]):
                e = reformed[statistic][k].squeeze()[1:]
                lower, centre, upper = upper_lower(e)
                x = np.arange(2, e.shape[0]+2)
                plt.plot(x, centre, linestyle='-', markersize=8, color=COLOURS[i], marker=MARKERS[i], linewidth=1.5)
                plt.fill_between(x, upper, lower, color=COLOURS[i] + [.1])
            plt.xlabel("Step", fontsize=22)
            plt.xticks(fontsize=16)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend([LABELS[k] for k in reformed[statistic].keys()], fontsize=16, frameon=False, loc=1, ncol=4)
            # frame = legend.get_frame()
            # frame.set_linewidth(S/)
            plt.yticks(fontsize=16)
            plt.ylabel(VALUE_LABELS[statistic], fontsize=22)
            # [i.set_linewidth(S/2) for i in plt.gca().spines.values()]
            # plt.gca().tick_params(width=S/2)
            if statistic != "Entropy":
                plt.yscale('log')
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
