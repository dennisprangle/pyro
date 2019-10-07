import argparse
import pickle
import numpy as np
import math

import torch
import matplotlib.pyplot as plt

output_dir = "./run_outputs/gradinfo/"


def main():

    designs = {}
    upper = {}
    lower = {}
    for name in ['docking-ace4', 'docking-baseline']:
        fname = output_dir + name + ".pickle"
        with open(fname, 'rb') as f:
            results = pickle.load(f)
            xi_history = results['xi_history']
            designs[name] = xi_history[-1, ...]
            upper[name] = results['final_upper_bound'][0]
            lower[name] = results['final_lower_bound'][0]

    designs['docking-ace4'] = designs['docking-ace4'][3,...]
    designs['docking-baseline'] = designs['docking-baseline'][0,...]

    d1 = designs['docking-ace4'].view(-1).numpy()
    d2 = designs['docking-baseline'].view(-1).numpy()

    bins = np.histogram(np.hstack((d1, d2)), bins=30)[1]  # get the bin edges

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(5,4))
    ax1.hist(d1, bins=bins)
    ax2.hist(d2, bins=bins)

    ax2.set_xlabel("Predicted binding affinity", fontsize=16)
    ax1.set_ylabel("#compounds", fontsize=14)
    ax2.set_ylabel("#compounds", fontsize=14)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    # ax1.set_yticks(fontsize=16)
    # ax2.set_yticks(fontsize=16)
    ax1.text(-97., 13., "ACE", ha='center', fontsize=16)
    ax2.text(-93., 13., "Expert", ha='center', fontsize=16)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result parser for design optimization (one shot)")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--sampling-interval", default=20, type=int)
    args = parser.parse_args()

    main()
