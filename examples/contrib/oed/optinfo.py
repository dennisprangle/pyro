from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.contrib.oed.eig import barber_agakov_ape, gibbs_y_eig, gibbs_y_eig_saddle, opt_mi
from pyro.contrib.oed.util import linear_model_ground_truth
from pyro.contrib.util import rmv, iter_plates_to_shape, lexpand, rvv, rexpand
from pyro.contrib.glmm import group_assignment_matrix

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


N = 15
prior_scale_tril = torch.tensor([[10., 0.], [0., 1./.55]])
AB_test_designs = torch.stack([group_assignment_matrix(torch.tensor([n, N-n])) for n in torch.linspace(0, N, N+1)])


def model_learn_xi(design_prototype):
    thetas = pyro.param('xi', torch.zeros(design_prototype.shape[-2]))
    xi = lexpand(torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1), 1)
    batch_shape = design_prototype.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        x = pyro.sample("x", dist.MultivariateNormal(lexpand(torch.zeros(2), *batch_shape),
                                                     scale_tril=lexpand(prior_scale_tril, *batch_shape)))
        prediction_mean = rmv(xi, x)
        return pyro.sample("y", dist.Normal(prediction_mean, torch.tensor(1.)).independent(1))


def model_fix_xi(design):
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        x = pyro.sample("x", dist.MultivariateNormal(lexpand(torch.zeros(2), *batch_shape),
                                                     scale_tril=lexpand(prior_scale_tril, *batch_shape)))
        prediction_mean = rmv(design, x)
        return pyro.sample("y", dist.Normal(prediction_mean, torch.tensor(1.)).independent(1))


model_fix_xi.w_sds = {"x": torch.tensor([10., 10.])}
model_fix_xi.obs_sd = torch.tensor(1.)


def make_posterior_guide(d):
    def posterior_guide(y_dict, design, observation_labels, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        A = pyro.param("A", torch.zeros(d, 2, N))
        scale_tril = pyro.param("scale_tril", lexpand(prior_scale_tril, d),
                                constraint=torch.distributions.constraints.lower_cholesky)
        mu = rmv(A, y)
        pyro.sample("x", dist.MultivariateNormal(mu, scale_tril=scale_tril))
    return posterior_guide


def make_marginal_guide(d):
    def marginal_guide(design, observation_labels, target_labels):
        mu = pyro.param("mu", torch.zeros(d, N))
        scale_tril = pyro.param("scale_tril", lexpand(torch.eye(N), d),
                                constraint=torch.distributions.constraints.lower_cholesky)
        pyro.sample("y", dist.MultivariateNormal(mu, scale_tril=scale_tril))
    return marginal_guide


def ilbo():
    # Here we estimate the APE (average posterior entropy) on a grid of possible xi's
    # We are computing the `posterior' lower bound
    pyro.clear_param_store()
    guide = make_posterior_guide(N + 1)
    barber_agakov_ape(model_fix_xi, AB_test_designs, "y", "x", guide=guide,
                      num_steps=3000, num_samples=10, optim=optim.Adam({"lr": 0.025}),
                      final_num_samples=500)
    ape_surf = barber_agakov_ape(model_fix_xi, AB_test_designs, "y", "x", guide=guide,
                                 num_steps=3000, num_samples=10, optim=optim.Adam({"lr": 0.0025}),
                                 final_num_samples=500)
    pyro.clear_param_store()
    guide = make_posterior_guide(1)
    # Here we optimize xi and phi simultaneously
    barber_agakov_ape(model_learn_xi, torch.zeros(1, N, 2), "y", "x", guide=guide,
                      num_steps=30000, num_samples=10, optim=optim.Adam({"lr": 0.01}),
                      final_num_samples=500)
    ape_star = barber_agakov_ape(model_learn_xi, torch.zeros(1, N, 2), "y", "x", guide=guide,
                                 num_steps=30000, num_samples=10, optim=optim.Adam({"lr": 0.001}),
                                 final_num_samples=500)
    thetas = pyro.param("xi")
    xi = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1)
    print("Grid of designs", AB_test_designs.sum(-2))
    print("APE on grid (not learning xi)", ape_surf)
    print("Learned xi_star", xi.abs().sum(0))
    print("APE from xi_star", ape_star)


def isaddle():
    pyro.clear_param_store()
    guide = make_marginal_guide(N + 1)
    true_eig_surf = linear_model_ground_truth(model_fix_xi, AB_test_designs, "y", "x")
    gibbs_y_eig(model_fix_xi, AB_test_designs, "y", "x", guide=guide,
                num_steps=3000, num_samples=10, optim=optim.Adam({"lr": 0.05}),
                final_num_samples=500)
    # eig_surf = gibbs_y_eig(model_fix_xi, AB_test_designs, "y", "x", guide=make_marginal_guide(N + 1),
    #                        num_steps=2000, num_samples=10, optim=optim.Adam({"lr": 0.005}),
    #                        final_num_samples=500)
    pyro.clear_param_store()
    guide = make_marginal_guide(1)

    for i in range(80):
        gibbs_y_eig(model_learn_xi, torch.zeros(1, N, 2), "y", "x", guide=guide,
                    num_steps=500, num_samples=10, optim=optim.Adam({"lr": 0.0025}),
                    exclude_names={"xi"})
        eig_star = gibbs_y_eig_saddle(model_learn_xi, torch.zeros(1, N, 2), "y", "x", guide=guide,
                                      num_steps=500, num_samples=10, optim=optim.Adam({"lr": 0.00075}),
                                      final_num_samples=500, adverse_names={"xi"})
    thetas = pyro.param("xi")
    xi = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1)
    print("Grid of designs", AB_test_designs.sum(-2))
    # print("EIG on grid (not learning xi)", eig_surf)
    print("True EIG on grid (not learning xi)", true_eig_surf)
    print("Learned xi_star", xi.abs().sum(0))
    print("EIG from xi_star", eig_star)


def unbiased_mi_gradient(xi):
    trace = pyro.poutine.trace(model_fix_xi).get_trace(xi)
    y = trace.nodes["y"]["value"]
    # trace.compute_log_prob()
    # lp = trace.nodes["y"]["log_prob"]

    #lp.backward()
    g1 = -(y - rmv(xi, trace.nodes["x"]["value"]))
    g2 = (rexpand(torch.eye(N), 2) * trace.nodes["x"]["value"]).transpose(-1, -2)
    trace2 = pyro.poutine.trace(pyro.condition(model_fix_xi, {"y": y})).get_trace(xi)
    g3 = (rexpand(torch.eye(N), 2) * trace2.nodes["x"]["value"]).transpose(-1, -2)

    return rvv(g1, (g2 - g3))


def opt_unbiased():
    thetas = torch.zeros(N, requires_grad=True)
    for i in range(5000):
        design = lexpand(torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1), 1)
        grad = unbiased_mi_gradient(design)
        final_grad = (grad * torch.stack([torch.cos(thetas), -torch.sin(thetas)], dim=-1)).sum(-1)
        thetas = thetas + 0.005 * final_grad
        print(design.abs().sum(1))


if __name__ == '__main__':
    # print("Saddle")
    # isaddle()
    # print("Lower bound")
    # ilbo()
    # opt_unbiased()
    opt_mi(model_learn_xi, torch.ones(1, N, 2), design_label="xi",
           observation_label="y", target_labels="x", num_samples=1,
           num_steps=5000, optim=optim.Adam({"lr": 0.005}))
