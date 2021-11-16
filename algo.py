
import numpy as np
from scipy.fftpack import ifftshift

import tools

def hio(y, mask, iter, beta=0.9, verbose=False):

    phase = np.random.rand(*y.shape) * 2 * np.pi
    x_prev = np.real(tools.ifft2d(y * np.exp(1j * phase)))

    for i in range(iter):

        if verbose:
            print('Iteration #{:d} '.format(i+1), end='')

        Fx_curr = y * np.exp(1j * phase)
        x_curr = np.real(tools.ifft2d(Fx_curr))
        idx = ((1+beta)*x_curr < x_prev) + (mask == False)
        # idx = (x_curr < 0) + (mask == False)
        x_curr[idx] = x_prev[idx] - beta * x_curr[idx]
        Fx_curr = tools.fft2d(x_curr)
        phase = np.angle(Fx_curr)
        x_prev = x_curr.copy()

        # residual = np.power(y - np.abs(new_f), 2).mean()

    return x_prev

def oss(y, mask, beta=0.9, verbose=False):

    phase = np.random.rand(*y.shape) * 2 * np.pi
    x_prev = np.real(tools.ifft2d(y * np.exp(1j * phase)))
    sigma1 = np.linspace(y.shape[0], 1. / y.shape[0], 10)
    sigma2 = np.linspace(y.shape[1], 1. / y.shape[1], 10)

    filtercount = 10
    iter = 2000
    Rsize = y.shape[0]
    X = np.arange(1, iter+1)
    FX=(filtercount+1-np.ceil(X*filtercount/iter))*np.ceil(iter/(1*filtercount));
    FX=((FX-np.ceil(iter/filtercount))*(2*Rsize)/np.max(FX))+(2*Rsize/10);

    for i in range(iter):

        if i == 0 or FX[i-1] != FX[i]:
            sigma = (FX[i], FX[i])
            filter = tools.get_gauss2d(y.shape, sigma, normalize=False)
            filter = ifftshift(filter)

        Fx_curr = y * np.exp(1j * phase)
        x_curr = np.real(tools.ifft2d(Fx_curr))
        idx = (x_curr < 0) + (mask == False)
        x_curr[idx] = x_prev[idx] - beta * x_curr[idx]
        Fx_curr = tools.fft2d(x_curr)
        idx = (mask == False)
        x_curr[idx] = np.real(tools.ifft2d(Fx_curr * filter))[idx]
        Fx_curr = tools.fft2d(x_curr)
        phase = np.angle(Fx_curr)
        x_prev = x_curr.copy()

    return x_curr

# find eig vector with max eig(A^* diag(y) A)
# normalize to lambda

def wf(y, mask, iter, verbose=False):

    m = y.size
    n = mask.sum()

    mu_max = 0.2
    t0 = 330.

    y = y**2
    lambd2 = y.sum()

    x = np.random.rand(*y.shape)
    x[~mask] = np.zeros(m-n)
    x = x / np.linalg.norm(x, 2)
    for i in range(1000):
        x = tools.ifft2d(y * tools.fft2d(x))
        x[~mask] = np.zeros(m-n)
        x = x / np.linalg.norm(x, 2)
    x = x * np.sqrt(lambd2)

    for t in range(iter):
        mu = np.min([1 - np.exp(-t/t0), mu_max])
        f = tools.fft2d(x)
        x = x - mu * tools.ifft2d((f**2 - y)*f) / lambd2
        x[~mask] = np.zeros(m-n)
        x = np.real(x)
    res = np.real(x)
    return res
