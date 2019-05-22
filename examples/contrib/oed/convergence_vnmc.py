from __future__ import absolute_import, division, print_function

import torch
import pickle
import math
import time
import argparse

import pyro
import pyro.optim as optim

from pyro.contrib.oed.eig import iwae_eig
from pyro.contrib.oed.util import linear_model_ground_truth
from pyro.contrib.util import lexpand
from pyro.contrib.glmm import group_assignment_matrix, known_covariance_linear_model
from pyro.contrib.glmm.guides import LinearModelPosteriorGuide


def main(fname, seed):
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)
    NPARALLEL = 25
    AB_test_1d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [6]])
    design = lexpand(AB_test_1d_10n_2p, NPARALLEL)
    model = known_covariance_linear_model(**{"coef_means": torch.tensor(0.),
                                             "coef_sds": torch.tensor([10., 1/.55]),
                                             "observation_sd": torch.tensor(1.)})

    lr = 0.05
    optimizer = optim.Adam({"lr": lr})
    print(linear_model_ground_truth(model, AB_test_1d_10n_2p, 'y', 'w'))

    for num_steps in [0, 125, 250, 500, 2500]:
        print("Num steps", num_steps)
        pyro.clear_param_store()
        guide = LinearModelPosteriorGuide(regressor_init=-10., scale_tril_init=torch.tensor([[10., 0.], [0., 1 / .55]]),
                                          d=(NPARALLEL, 1), w_sizes=model.w_sizes, y_sizes={"y": 10})
        t = time.time()
        iwae_eig(model, design, "y", "w", num_samples=(10, 1), num_steps=num_steps, guide=guide, optim=optimizer)
        t1 = time.time() - t
        for M in torch.logspace(math.log10(10), math.log10(151), 10):
            M = int(M)
            print("M", M)
            t = time.time()
            eig_surface_iwae = iwae_eig(model, design, "y", "w", num_samples=(1, 1), final_num_samples=(M*M, M),
                                        num_steps=0, guide=guide, optim=optimizer)
            elapsed = t1 + time.time() - t
            print(eig_surface_iwae)

            results = {"num_steps": num_steps, "M": M, "surface": eig_surface_iwae, "elapsed": elapsed,
                       "lr": lr, "seed": seed}
            with open('run_outputs/{}.result_stream.pickle'.format(fname), 'ab') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence example for VNMC")
    parser.add_argument("--fname", nargs="?", default="", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    args = parser.parse_args()
    main(args.fname, args.seed)
