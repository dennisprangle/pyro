from __future__ import absolute_import, division, print_function

import torch
import pickle
import math

import pyro
import pyro.optim as optim

from pyro.contrib.oed.eig import iwae_eig
from pyro.contrib.util import lexpand
from pyro.contrib.glmm import group_assignment_matrix, known_covariance_linear_model
from pyro.contrib.glmm.guides import LinearModelPosteriorGuide


if __name__ == '__main__':
    AB_test_1d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [6]])
    design = lexpand(AB_test_1d_10n_2p, 10)
    model = known_covariance_linear_model(**{"coef_means": torch.tensor(0.),
                                             "coef_sds": torch.tensor([10., 1/.55]),
                                             "observation_sd": torch.tensor(1.)})

    optim = optim.Adam({"lr": 0.05})

    for num_steps in [0, 125, 250]:
        print("Num steps", num_steps)
        pyro.clear_param_store()
        guide = LinearModelPosteriorGuide(tikhonov_init=10., scale_tril_init=torch.tensor([[10., 0.], [0., 1 / .55]]),
                                          d=(10, 1), w_sizes=model.w_sizes)
        iwae_eig(model, design, "y", "w", num_samples=(10, 1), num_steps=num_steps, guide=guide, optim=optim)
        for M in torch.logspace(0, math.log10(101), 10):
            M = int(M)
            print("M", M)
            eig_surface_iwae = iwae_eig(model, design, "y", "w", num_samples=(1, 1), final_num_samples=(M*M, M),
                                        num_steps=0, guide=guide, optim=optim)

            results = {"num_steps": num_steps, "M": M, "surface": eig_surface_iwae}
            with open('run_outputs/nmc_iwae.result_stream.pickle', 'ab') as f:
                pickle.dump(results, f)
