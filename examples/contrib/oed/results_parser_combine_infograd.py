import argparse
import pickle
import math

import torch
import matplotlib.pyplot as plt

output_dir = "./run_outputs/gradinfo/"


def main(names, sampling_interval):

    combined = {}
    for name in names.split(","):
        fname = output_dir + name + ".pickle"
        with open(fname, 'rb') as f:
            results = pickle.load(f)
        combined[name] = results

    legend = []
    for name in names.split(","):
        wall_time = combined[name]['wall_times'].detach().numpy()[::sampling_interval]
        est_hist = combined[name]['est_eig_history'].detach().numpy()[::sampling_interval]
        hist = combined[name]['eig_history'].detach().numpy()[::sampling_interval]
        mean, se = hist.mean(1), hist.std(1)/math.sqrt(hist.shape[1])
        #plt.plot(wall_time, est_hist)
        plt.plot(wall_time, mean)
        plt.fill_between(wall_time, mean - se, mean + se, alpha=0.1)
        legend.extend([combined[name]['estimator'] + ' exact'])
    plt.legend(legend)
    plt.show()

    for name in names.split(","):
        print(name, combined[name]['eig_history'][-1,...].mean(),
              combined[name]['eig_history'][-1,...].std()/math.sqrt(combined[name]['eig_history'].shape[1]),
              combined[name]['wall_times'][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result parser for design optimization (one shot)")
    parser.add_argument("--names", default="", type=str)
    parser.add_argument("--sampling-interval", default=1, type=int)
    args = parser.parse_args()

    main(args.names, args.sampling_interval)
