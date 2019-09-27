import argparse
import pickle
import numpy as np

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
        #plt.plot(wall_time, est_hist)
        plt.plot(wall_time, hist)
        legend.extend([combined[name]['estimator'] + ' exact'])
    plt.legend(legend)
    plt.show()

    for name in names.split(","):
        print(name, combined[name]['xi_history'][-1,...], combined[name]['eig_history'][-1], combined[name]['wall_times'][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result parser for design optimization (one shot)")
    parser.add_argument("--names", default="", type=str)
    parser.add_argument("--sampling-interval", default=20, type=int)
    args = parser.parse_args()

    main(args.names, args.sampling_interval)
