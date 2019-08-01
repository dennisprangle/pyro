from contextlib import ExitStack
import math

import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.contrib.oed.eig import posterior_eig
from pyro.contrib.oed.util import linear_model_ground_truth
from pyro.contrib.util import rmv, iter_plates_to_shape, lexpand, rvv, rexpand


N = 2
prior_scale_tril = torch.tensor([[10., 0.], [0., 2.]])
xi_init = (math.pi/3) * torch.ones(N,)


def model_learn_xi(design_prototype):
    thetas = pyro.param('xi', xi_init)
    xi = lexpand(torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1), 1)
    batch_shape = design_prototype.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(2), scale_tril=prior_scale_tril))
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


model_fix_xi.w_sds = {"x": prior_scale_tril.diagonal()}
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


if __name__ == '__main__':
    pyro.clear_param_store()

    guide = make_posterior_guide(1)

    history = [xi_init.clone()]
    estimated_eig_history = [torch.tensor([0.])]
    lr = 0.05
    target_lr = 0.005
    n_steps = 20
    scale = (target_lr/lr)**(1/n_steps)
    for i in range(n_steps):
        eig_hat = posterior_eig(model_learn_xi, torch.zeros(1, N, 2), "y", "x", guide=guide,
                                num_steps=100, num_samples=10, optim=optim.Adam({"lr": lr}))
        xi = pyro.param("xi").detach().clone()
        history.append(xi)
        estimated_eig_history.append(eig_hat)
        lr *= scale

    history = torch.stack(history, dim=0)
    estimated_eig_history = torch.cat(estimated_eig_history)
    eig_history = linear_model_ground_truth(model_fix_xi, torch.stack([torch.sin(history), torch.cos(history)], dim=-1),
                                            "y", "x")

    import matplotlib.pyplot as plt

    D = 100
    b0low = min(0, history[:, 0].min()) - 0.1
    b0up = max(math.pi, history[:, 0].max()) + 0.1
    b1low = min(0, history[:, 1].min()) - 0.1
    b1up = max(math.pi, history[:, 1].max()) + 0.1
    theta1 = torch.linspace(b0low, b0up, D)  # D
    theta2 = torch.linspace(b1low, b1up, D)  # D
    d1 = torch.stack([torch.sin(theta1), torch.cos(theta1)], dim=-1).unsqueeze(-2).unsqueeze(1).expand(D, D, 1, 2)
    d2 = lexpand(torch.stack([torch.sin(theta2), torch.cos(theta2)], dim=-1).unsqueeze(-2), D)  # D, D, 1, 2
    d = torch.cat([d1, d2], dim=-2)
    true_eig = linear_model_ground_truth(model_fix_xi, d, "y", "x")

    plt.imshow(true_eig, cmap="gray", extent=[b0low, b0up, b1low, b1up], origin='lower')
    plt.scatter(history[:, 0], history[:, 1], c=torch.arange(history.shape[0]), marker='x', cmap='summer')
    plt.show()

    plt.plot(eig_history.detach().numpy())
    plt.plot(estimated_eig_history.detach().numpy())
    plt.show()
