import argparse
import pickle
import numpy as np

import torch
import matplotlib.pyplot as plt

output_dir = "./run_outputs/gradinfo/"


def ma(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def main(name, sampling_interval):

    fname = output_dir + name + ".pickle"
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    xi_history = results['xi_history']
    est_eig_history = results['est_eig_history']
    eig_history = results.get('eig_history')
    eig_lower = results.get('lower_history')
    eig_upper = results.get('upper_history')


    plt.plot(est_eig_history[::1000].clamp(min=0, max=2).numpy())
    plt.plot(torch.cat(eig_lower).clamp(min=0, max=2).numpy())
    plt.plot(torch.cat(eig_upper).clamp(min=0, max=2).numpy())
    print("last upper", eig_upper[-1], "last lower", eig_lower[-1])

    plt.legend(["Loss", "Lower bound", "Upper bound"])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result parser for design optimization (one shot) with a linear model")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--sampling-interval", default=1000, type=int)
    args = parser.parse_args()

    main(args.name, args.sampling_interval)
