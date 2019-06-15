from __future__ import absolute_import, division, print_function
import torch
import math
import pyro
from pyro.optim import Adam
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
from pyro.distributions import Normal, Bernoulli
from torch.distributions.constraints import positive
from pyro.contrib.util import iter_plates_to_shape

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


mean_prior = 0.30
sigma_prior = 0.05
entropy = 0.5 * math.log(2.0 * math.pi * math.e * (sigma_prior ** 2))
print("prior_entropy", entropy)


def model(design):
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(design.shape):
            stack.enter_context(plate)
        theta = pyro.sample("theta", Normal(mean_prior * torch.ones(design.shape),
                                            sigma_prior * torch.ones(design.shape)))
        guess_prob = torch.sigmoid(-theta * design)
        return pyro.sample("y", Bernoulli(guess_prob))


def guide(design, observation_labels, target_labels):
    p_logit = pyro.param("p_logit", mean_prior * torch.ones(design.shape[-1]))
    pyro.sample("y", Bernoulli(logits=p_logit))


design = torch.arange(20).float()


# Ballpark
estimated_eig = marginal_eig(model, design, "y", "theta", num_samples=20,
                             num_steps=1000, guide=guide,
                             optim=Adam({"lr": 0.25}))
# Finesse
estimated_eig = marginal_eig(model, design, "y", "theta", num_samples=20,
                             num_steps=1000, guide=guide,
                             optim=Adam({"lr": 0.001}), final_num_samples=10000)

nmc_eig = nmc_eig(model, design, "y", "theta", N=120*120, M=120)

print("estimated_eig", estimated_eig.data.numpy())
print("guide p_logit", pyro.param("p_logit").data.numpy())
print("nmc_eig", nmc_eig.data.numpy())
print("nmc < marginal", (nmc_eig < estimated_eig).numpy())
