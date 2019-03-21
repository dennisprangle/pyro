import torch
import argparse
import datetime
import pickle

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.glmm.glmm import sigmoid_location_model
from pyro.contrib.glmm.guides import SigmoidMarginalGuide
from pyro.contrib.util import iter_plates_to_shape, lexpand
from pyro.contrib.oed.eig import gibbs_y_eig, elbo_learn

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


def prior_factory(mean, sd):
    def f(design):
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            loc_shape = batch_shape + (design.shape[-1],)
            pyro.sample("loc", dist.Normal(mean.expand(loc_shape),
                                           sd.expand(loc_shape)).to_event(1))
    return f


def elboguide(design, dim=10):
    mean = pyro.param("mean", lexpand(torch.tensor(0.), dim, 1, 1))
    sd = pyro.param("sd", lexpand(torch.tensor(50.), dim, 1, 1))
    f = prior_factory(mean, sd)
    return f(design)


def main(num_steps, num_parallel, experiment_name):
    output_dir = "./run_outputs/location/"
    if not experiment_name:
        experiment_name = output_dir+"{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir+experiment_name
    results_file = experiment_name + '.result_stream.pickle'
    typs = ['posterior_mean']
    true_model = sigmoid_location_model(lexpand(torch.tensor(44.6), num_parallel, 1, 1), torch.tensor([0.]), torch.tensor([1.]),
                                        torch.tensor(5.))
    for typ in typs:
        pyro.clear_param_store()
        marginal_mu_init, marginal_sigma_init = 0., 50.
        oed_n_samples, oed_n_steps, oed_final_n_samples, oed_lr = 10, 2000, 2000, 0.04
        elbo_n_samples, elbo_n_steps, elbo_lr = 10, 2000, 0.04

        guide = SigmoidMarginalGuide(d=(200,), y_sizes={"y": 1}, mu_init=marginal_mu_init, sigma_init=marginal_sigma_init)

        prior = sigmoid_location_model(lexpand(torch.tensor(0.), num_parallel, 1, 1), torch.tensor(50.), torch.tensor([1.]), torch.tensor(5.))
        mean, sd = lexpand(torch.tensor(0.), num_parallel, 1, 1), lexpand(torch.tensor(50.), num_parallel, 1, 1)

        d_star_designs = torch.tensor([])
        ys = torch.tensor([])

        for step in range(num_steps):
            small_design = torch.linspace(0., 100., 200).unsqueeze(-1).unsqueeze(-1)
            design = lexpand(small_design, num_parallel)
            model = sigmoid_location_model(mean, sd, torch.tensor([1.]), torch.tensor(5.))
            results = {'typ': typ, 'step': step}
            if typ == 'oed':
                # Throws ArithmeticError if NaN encountered
                estimation_surface = gibbs_y_eig(
                    model, design, "y", "loc",
                    oed_n_samples, oed_n_steps, guide,
                    optim.Adam({"lr": oed_lr}), False, None, oed_final_n_samples
                )
                results['estimation_surface'] = estimation_surface
                print("EIG surface", estimation_surface)
                print(estimation_surface.shape)

                d_star_index = torch.argmax(estimation_surface, dim=1)
                print(d_star_index.shape)
                d_star_design = small_design[d_star_index, ...].unsqueeze(1)
                print(d_star_design.shape)
            elif typ == 'posterior_mean':
                d_star_design = mean.unsqueeze(1)

            results['d_star_design'] = d_star_design
            print('design', d_star_design)
            d_star_designs = torch.cat([d_star_designs, d_star_design], dim=-2)
            y = true_model(d_star_design)
            ys = torch.cat([ys, y], dim=-1)
            print(ys, ys.shape)
            results['y'] = y

            elbo_learn(
                prior, d_star_designs, ["y"], ["loc"], elbo_n_samples, elbo_n_steps,
                elboguide, {"y": ys}, optim.Adam({"lr": elbo_lr})
            )
            print("mean", mean, "sd", sd)
            results['mean'] = mean
            results['sd'] = sd
            mean = pyro.param("mean").detach().data.clone()
            sd = pyro.param("sd").detach().data.clone()
            print("mean", mean, "sd", sd)

            with open(results_file, 'ab') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design")
    parser.add_argument("--num-steps", nargs="?", default=25, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    args = parser.parse_args()
    main(args.num_steps, args.num_parallel, args.name)
