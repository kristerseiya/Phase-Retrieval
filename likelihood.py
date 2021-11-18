
import numpy as np
import scipy
import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
import pyro
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
            - 0.5 * (value - self.loc).square() / s2
            + torch.special.i0e(value * self.loc / s2).log()
        ).sum()

class FourierMagGaussian:
    def __init__(self, y, alpha, mask):
        self.y = y
        self.alpha = alpha
        self.mask = mask

    def __call__(self, x):
        f = tools.fft2d(x)
        return np.real(self.alpha * np.sum((y - np.abs(f))**2))

    def grad(self, x):
        f = tools.fft2d(x)
        return self.alpha * np.real(tools.ifft2d(f - y * f / np.abs(f)))

    def prox(self, x, tau=1):
        z = x
        zf = tools.fft2d(z)
        mag = 1 / (self.alpha + tau) * (self.alpha * self.y + tau * np.abs(zf))
        res = mag * np.exp(1j*np.angle(zf))
        res = np.real(tools.ifft2d(res))
        res = tools.ifft2d(res)
        return res

class FourierMagRician:
    def __init__(self, y, lik_scale, pri_scale):
        self.y = y
        self.n_scale = lik_scale
        self.t_scale = pri_scale

    def _posterior(self, v):
        x = pyro.sample("x", Rician(v, self.t_scale).to_event(2))
        pyro.sample("y", Rician(x, self.n_scale).to_event(2), obs=self.y)

    def _guide(self, v):
        x_map = pyro.param("x_map",
                           (self.n_scale*v + self.t_scale*self.y) / (self.n_scale + self.t_scale),
                           constraint=constraints.positive)
        pyro.sample("x", dist.Delta(x_map, event_dim=2))

    def prox(self, x, t=1, lr=1e-2, n_step=51):

        fx = torch.fft.fft2(x, norm='ortho')
        pyro.clear_param_store()
        adam = pyro.optim.Adam({"lr": lr})
        svi = SVI(self._posterior, self._guide, adam, loss=Trace_ELBO())
        n_steps = 51
        for step in range(n_steps):
            loss = svi.step(torch.abs(fx))

        mag = pyro.param('x_map').detach()
        phs = torch.angle(fx)
        return torch.real(torch.fft.ifft2(mag*torch.exp(1j*phs), norm='ortho'))
