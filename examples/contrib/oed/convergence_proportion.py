from __future__ import absolute_import, division, print_function

import torch
import pickle
import math
import argparse
import time

import pyro
import pyro.optim as optim

from pyro.contrib.oed.eig import gibbs_y_eig
from pyro.contrib.oed.util import ba_eig_lm
from pyro.contrib.util import lexpand
from pyro.contrib.glmm import group_assignment_matrix, known_covariance_linear_model
from pyro.contrib.glmm.guides import LinearModelPosteriorGuide, NormalMarginalGuide


def main(fname, seed):
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)
    NPARALLEL = 100
    AB_test_1d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [6]])
    design = lexpand(AB_test_1d_10n_2p, NPARALLEL)
    model = known_covariance_linear_model(**{"coef_means": torch.tensor(0.),
                                             "coef_sds": torch.tensor([10., 1/.55]),
                                             "observation_sd": torch.tensor(1.)})

    for Ti, T in enumerate(torch.logspace(math.log10(50), math.log10(5000), 10)):
        T = int(T)
        print("T", T)
        pyro.clear_param_store()
        lr = min(1./math.sqrt(T), 1.)
        #lr=0.01
        print(lr)
        optimizer = optim.Adam({"lr": lr})
        guide = LinearModelPosteriorGuide(regressor_init=-10., scale_tril_init=torch.tensor([[10., 0.], [0., 1 / .55]]),
                                          d=(NPARALLEL, 1), w_sizes=model.w_sizes, y_sizes={"y": 10})
        t = time.time()
        ba_eig_lm(model, design, "y", "w", num_samples=1, final_num_samples=1,
                  num_steps=T, guide=guide, optim=optimizer)
        t1 = time.time() - t
        t = time.time()
        eig_surface = ba_eig_lm(model, design, "y", "w", num_samples=1, final_num_samples=T,
                                num_steps=0, guide=guide, optim=optimizer)
        elapsed = t1 + time.time() - t
        results = {"method": "posterior", "T": T, "Ti": Ti, "surface": eig_surface, "elapsed": elapsed, "N": T,
                   "Ni": Ti, "seed": seed, 'lr': lr}
        with open('run_outputs/{}.result_stream.pickle'.format(fname), 'ab') as f:
            pickle.dump(results, f)

    for Ti, T in enumerate(torch.logspace(math.log10(50), math.log10(5000), 10)):
        T = int(T)
        print("T", T)
        pyro.clear_param_store()
        lr = min(2.5/math.sqrt(T), 1.)
        # lr=0.05
        # print(lr)
        optimizer = optim.Adam({"lr": lr})
        guide = NormalMarginalGuide(d=(NPARALLEL, 1), y_sizes={"y": 10}, sigma_init=3.)
        t = time.time()
        gibbs_y_eig(model, design, "y", "w", 1, num_steps=T, guide=guide, optim=optimizer, final_num_samples=1)
        t1 = time.time() - t
        t = time.time()
        eig_surface = gibbs_y_eig(model, design, "y", "w", 1, num_steps=0, guide=guide, optim=optimizer,
                                  final_num_samples=T)
        elapsed = t1 + time.time() - t
        results = {"method": "marginal", "T": T, "Ti": Ti, "surface": eig_surface, "elapsed": elapsed, "N": T,
                   "Ni": Ti, "seed": seed, 'lr': lr}
        with open('run_outputs/{}.result_stream.pickle'.format(fname), 'ab') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence example for posterior, marginal")
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--fname", nargs="?", default="", type=str)
    args = parser.parse_args()
    main(args.fname, args.seed)
