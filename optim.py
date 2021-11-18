
import numpy as np
import copy

class ADMM:
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def init(self, v, u):
        self.v_init = copy.deepcopy(v)
        self.u_init = copy.deepcopy(u)

    def run(self, iter=100, relax=0., return_value='both', verbose=False):

        v = self.v_init
        u = self.u_init

        for i in range(iter):

            if verbose:
                print('Iteration #{:d}'.format(i+1))

            x = self.f.prox(v-u)

            xr = (1 - relax) * x + relax * v
            xu = xr + u

            v = self.g.prox(xu)

            diff = xr - v
            u = u + diff

        if return_value == 'both':
            return x, v
        elif return_value == 'x':
            return x
        return v

class RED:
    def __init__(self, fidelity, denoiser, lambd):
        self.fidelity = fidelity
        self.denoiser = denoiser
        self.lambd = lambd

    def init(self, x):
        self.x0 = x

    def run(self, iter, method='gd', step=None, beta=None):

        if method == 'gd':

            x = self.x0
            for i in range(iter):
                print('Iteration #{:d}'.format(i+1), end='')
                x = x - step * (self.fidelity.grad(x) + self.lambd * (x - self.denoiser(x)))
                loss1 = self.fidelity(x)
                loss2 = self.lambd / 2 * (x * (x - self.denoiser(x))).sum()
                print('   Loss: {:.5f}, {:.5f}, {:.5f}'.format(loss1+loss2, loss1, loss2))

        elif method == 'admm':

            v = self.x0
            u = np.zeros_like(v)
            for i in range(iter):
                print('Iteration #{:d}'.format(i+1), end='')
                x = self.fidelity.prox(v-u)
                for k in range(1):
                    v = 1 / (beta + self.lambd) * (self.lambd * self.denoiser(v) + beta * (x + u))
                    # v = np.clip(v, 0, 1)
                u = u + x - v
                loss1 = self.fidelity(x)
                loss2 = (x * (x - self.denoiser(x))).sum()
                print('   Loss: {:.5f}, {:.5f}, {:.5f}'.format(loss1+self.lambd/2*loss2, loss1, loss2))

        elif method == 'fasta':

            x1 = self.x0
            beta = 1
            tau = 1e-7
            shrink = 0.5
            f1 = self.fidelity(x1)
            for i in range(iter):
                print('Iteration #{:d}'.format(i+1), end='')
                x0 = x1
                gradf0 = self.fidelity.grad(x0)
                x1 = x0 - tau * gradf0
                x1 = np.real(x1)
                x1 = 1 / (beta + self.lambd) * (tau * self.lambd * self.denoiser(x1) + beta * (x1))
                f0 = f1
                f1 = self.fidelity(x1)
                dx = x1 - x0
                backtrack_count = 0
                while (f1 - 1e-12 > f0 + np.real(np.sum(dx*gradf0)) + 1 / 2 / tau * np.sum(dx**2)) and backtrack_count < 20:
                    tau = tau * shrink
                    x1 = x0 - tau * gradf0
                    x1 = np.real(x1)
                    x1 = 1 / (beta + self.lambd) * (tau * self.lambd * self.denoiser(x1) + beta * (x1))
                    f1 = self.fidelity(x1)
                    dx = x1 - x0
                    backtrack_count = backtrack_count + 1
                g1 = (x1 * (x1 - self.denoiser(x1))).sum()
                print('   Loss: {:.5f}, {:.5f}, {:.5f}, {:d}'.format(f1+self.lambd*g1, f1, self.lambd*g1, backtrack_count))

        return x1
