
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import tools
# import dnsr
# from DnCNN import load_dncnn
import torch
import torch.special
import pyro
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from pyro.distributions import TorchDistribution
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

class Rician(TorchDistribution):
    arg_constraints = {"loc": constraints.positive,
                       "scale": constraints.positive}
    support = constraints.positive

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def log_prob(self, value):
        s2 = self.scale.square()
        return (
            value.log() - s2.log()
            - 0.5 * (value.square() + self.loc.square()) / s2
            + torch.special.i0(value * self.loc / s2).log()
        ).sum()

size = (1, 1, 2, 2)
data = torch.rand(size)
v = torch.rand(size)
print(v)
print(data)

def rician_model(v):
    # x = pyro.sample("x", dist.ImproperUniform(constraints.positive, (), (size)))
    x = pyro.sample("x", Rician(v, 1).to_event(2))
    pyro.sample("y", Rician(x, 1).to_event(2), obs=data)

def guide_map(v):
    x_map = pyro.param("x_map", v.clone(),
                       constraint=constraints.positive)
    pyro.sample("x", dist.Delta(x_map, event_dim=2))

def train(model, guide, lr=0.01):
    pyro.clear_param_store()
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    n_steps = 51
    for step in range(n_steps):
        loss = svi.step(v)
        if step % 50 == 0:
            print('[iter {}]  loss: {:.4f}'.format(step, loss))

train(rician_model, guide_map)
# print("Our MAP estimate of the latent fairness is {:.3f}".format(
#       pyro.param("f_map").item()))
print(pyro.param("x_map"))
