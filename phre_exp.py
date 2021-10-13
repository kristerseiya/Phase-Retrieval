
import numpy as np
from PIL import Image
from scipy.fftpack import fft, ifft, ifftshift, fftshift
import argparse
import scipy.io

import algo
import tools
import dnsr
import optim

parser = argparse.ArgumentParser()
parser.add_argument('--lambd', type=float, default=1)
parser.add_argument('--hioiter', type=int, default=50)
parser.add_argument('--pnpiter', type=int, default=600)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--save', default='recon.png')
parser.add_argument('--supp', nargs=2, type=int, required=True)
parser.add_argument('--scale', type=float, default=1)
args = parser.parse_args()

supdim = args.supp
pnpiter1 = args.pnpiter - 2 * args.pnpiter // 3
pnpiter2 = args.pnpiter // 3
pnpiter3 = args.pnpiter // 3

class FourMagMSE:
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
        # return self.alpha * np.conjugate(tools.fft2d(np.conjugate(f) - self.y * np.conjugate(f) / np.abs(f)))

    def prox(self, x, tau=1):
        xf = tools.fft2d(x)
        mag = 1 / (self.alpha + tau) * (self.alpha * self.y + tau * np.abs(xf))
        res = mag * np.exp(1j*np.angle(xf))
        res = np.real(tools.ifft2d(res))
        return res

y = scipy.io.loadmat('simulated_circle.mat')
y = y['simulated_circle']
y = ifftshift(y) * np.sqrt(y.size)
# y = y * np.sqrt(y.size)
# y = scipy.io.loadmat('15mm_circle_082621.mat')
# y = ifftshift(y) / 10.
# y = scipy.io.loadmat('LUX_082621.mat')
# y = y['For_PR']
# y = y['Pi_spec_sim_abs_PR']
# y = ifftshift(y) * 3.
y = y * args.scale

v00 = np.real(tools.ifft2d(y))

# print(tools.ifft2d(y))
#  * np.sqrt(y.size)
mask = np.ones(y.shape, dtype=bool) * False
midpt = (mask.shape[0]//2, mask.shape[1]//2)
mask[midpt[0]-supdim[0]//2:midpt[0]+supdim[0]-supdim[0]//2,
     midpt[1]-supdim[1]//2:midpt[1]+supdim[1]-supdim[1]//2] = np.ones(supdim, dtype=bool) * True

v0 = algo.hio(y, mask, args.hioiter)
v0[~mask] = np.zeros(y.size - mask.sum())

fidelity = FourMagMSE(y, 50**2 / (args.lambd**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn50_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v0, np.zeros(y.shape))
v1 = optimizer.run(mask, supdim, iter=pnpiter1, return_value='v', verbose=False, beta=args.beta)

fidelity = FourMagMSE(y, 25**2 / (args.lambd**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn25_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v1, np.zeros(y.shape))
v2 = optimizer.run(mask, supdim, iter=pnpiter2, return_value='v', verbose=False, beta=args.beta)

fidelity = FourMagMSE(y, 10**2 / (args.lambd**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn10_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v2, np.zeros(y.shape))
v3 = optimizer.run(mask, supdim, iter=pnpiter3, return_value='v', verbose=False, beta=args.beta)

v0 = v0[mask].reshape(supdim)
v1 = v1[mask].reshape(supdim)
v2 = v2[mask].reshape(supdim)
v3 = v3[mask].reshape(supdim)

res = tools.stackview([v0, v1, v2, v3], method='Pillow')

res = np.clip(v3 * 255, 0, 255)
res = Image.fromarray(res.astype(np.uint8))
res.save(args.save, format='PNG')

# v0 = v0[mask].reshape(supdim)
# tools.stackview([v0], method='Pillow')

# res = np.clip(v3 * 255, 0, 255)
# res = Image.fromarray(res.astype(np.uint8))
# res.save(args.save, format='PNG')
