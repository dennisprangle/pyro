import argparse
import pickle
import numpy as np

import torch
import matplotlib.pyplot as plt

output_dir = "./run_outputs/gradinfo/"


def main(name, sampling_interval):

    fname = output_dir + name + ".pickle"
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    xi_history = results['xi_history']
    est_eig_history = results['est_eig_history']
    eig_history = results.get('eig_history')
    eig_heatmap = results.get('eig_heatmap')
    heatmap_extent = results.get('extent')
    eig_lower = results.get('lower_history')
    eig_upper = results.get('upper_history')

    if xi_history.shape[-1] <= 2:
        if eig_heatmap is not None:
            plt.imshow(eig_heatmap, cmap="gray", extent=heatmap_extent, origin='lower')
        x, y = xi_history[::sampling_interval, 0].detach(), xi_history[::sampling_interval, 1].detach()
        plt.scatter(x, y, c=torch.arange(x.shape[0]), marker='x', cmap='summer')
        plt.show()
    elif xi_history.shape[-1] > 50:
        plt.hist(xi_history[-1].numpy(), bins=30)
        plt.show()
    else:
        print(xi_history[-1, ...])

    if eig_upper is not None and eig_lower is not None:
        plt.plot(eig_lower.clamp(min=0, max=2).numpy())
        plt.plot(eig_upper.clamp(min=0, max=2).numpy())
        print("last upper", eig_upper[-1], "last lower", eig_lower[-1])

        plt.legend(["Lower bound", "Upper bound"])
        plt.show()

    plt.plot(est_eig_history.detach().clamp(min=0).numpy()[::sampling_interval])
    if eig_history is not None:
        plt.plot(eig_history.detach().numpy()[::sampling_interval])
        print("Final true EIG", eig_history[-1].item())
        if eig_heatmap is not None:
            print("Max EIG over surface", eig_heatmap.max().item())
            print("Discrepancy", (eig_heatmap.max() - eig_history[-1]).item())
        plt.legend(["Approximate EIG", "True EIG"])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result parser for design optimization (one shot)")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--sampling-interval", default=20, type=int)
    args = parser.parse_args()

    main(args.name, args.sampling_interval)
