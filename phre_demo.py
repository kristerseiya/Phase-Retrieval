
import numpy as np
from PIL import Image
from scipy.fftpack import fft, ifft, fftshift
import argparse

import optim
import algo
import tools
from Denoisers import dnsr

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='path to image', type=str, required=True)
parser.add_argument('--hioiter', help='number of iterations of HIO (initialization)', type=int, default=50)
parser.add_argument('--sigma', nargs='+', type=int, required=True)
parser.add_argument('--pnpiter', help='total number of iterations with PnP-ADMM',
                                nargs='+', type=int, required=True)
parser.add_argument('--noise', help='noise type (\'gaussian\' or \'poisson\' or \'rician\')', type=str, default='gaussian')
parser.add_argument('--noiselvl', help='standard deviation if noise type is gaussian,'
                            'if poisson, the product of this parameter and the signal'
                            'would be used as a standard deviation of gaussian to'
                            'simulate a poisson noise', type=float, default=1)
parser.add_argument('--samprate', help='oversampling rate for the measurement', type=float, default=4)
parser.add_argument('--beta', help='parameter for HIO', type=float, default=0.9)
parser.add_argument('--display', help='display result if given', action='store_true')
parser.add_argument('--save', help='save result if given a filename', type=str, default=None)
args = parser.parse_args()

if len(args.pnpiter) != len(args.sigma):
    raise ValueError('sheeit')
# pnpiter1 = args.pnpiter - 2 * args.pnpiter // 3
# pnpiter2 = args.pnpiter // 3
# pnpiter3 = args.pnpiter // 3

img = Image.open(args.image).convert('L')
img = np.array(img) / 255.

pad_len_1 = int(img.shape[0] * (np.sqrt(args.samprate) - 1)) // 2
pad_len_2 = int(img.shape[1] * (np.sqrt(args.samprate) - 1)) // 2
imgpad = np.pad(img, ((pad_len_1, pad_len_1), (pad_len_2, pad_len_2)), 'constant', constant_values=((0, 0), (0, 0)))
mask = np.ones(imgpad.shape, dtype=bool) * False
mask[pad_len_1:-pad_len_1, pad_len_2:-pad_len_2] = True
n = img.size
m = imgpad.size

if args.noise == 'gaussian':
    y = np.real(np.abs(tools.fft2d(imgpad))) + np.random.normal(size=imgpad.shape) * args.noiselvl / 255.
    sigma = args.noiselvl if args.noiselvl > 0 else 1

elif args.noise == 'poisson':
    yy = np.real(np.abs(tools.fft2d(imgpad)))
    alpha = args.noiselvl / 255.
    intensity_noise = alpha * yy * np.random.normal(size=imgpad.shape)
    y = (yy**2 + intensity_noise)
    y = y * (y > 0)
    y = np.sqrt(y)
    sigma = np.std(y - yy) * 255.

elif args.noise == 'rician':
    yy = tools.fft2d(imgpad)
    y = yy + (np.random.normal(size=imgpad.shape) + 1j * np.random.normal(size=imgpad.shape)) * args.noiselvl / 255. / np.sqrt(2)
    y = np.abs(y)
    sigma = np.std(y - np.abs(yy))*255

# x = np.random.rand(*y.shape)
# x[~mask] = np.zeros(m-n)
# x = x / np.linalg.norm(x, 2)
# y2 = y**2
# for i in range(1000):
#     x = np.real(tools.ifft2d(y2 * tools.fft2d(x)))
#     x[~mask] = np.zeros(m-n)
#     x = x / np.linalg.norm(x, 2)
# v = x
v = algo.hio(y, mask, args.hioiter, beta=args.beta, verbose=False)
# v[~mask] = np.zeros(img.size - mask.sum())

v_arr = []
v_arr += [v]


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

    def prox(self, x, tau=1):
        z = x
        zf = tools.fft2d(z)
        mag = 1 / (self.alpha + tau) * (self.alpha * self.y + tau * np.abs(zf))
        res = mag * np.exp(1j*np.angle(zf))
        res = np.real(tools.ifft2d(res))
        return res

class SpaceCnstrat:
    def __init__(self, mask, supdim):
        self.mask = mask
        self.supdim = supdim
        self.denoiser = dnsr.cDnCNN('Denoisers/dnsr/DnCNN/weights/cDnCNNv6_1-50.pth')

    def prox(self, x):
        v = np.zeros(x.shape)
        x = (x[self.mask]).reshape(self.supdim)

        v[self.mask] = self.denoiser(x).flatten()

        idx = (v < 0) * self.mask
        v[idx] = np.zeros(idx.sum())
        return v

spcnst = SpaceCnstrat(mask, img.shape)

for it, a in zip(args.pnpiter, args.sigma):
    fidelity = FourMagMSE(y, a**2 / (sigma**2), mask)
    spcnst.denoiser.set_param(a / 255.)
    optimizer = optim.ADMM(fidelity, spcnst)
    optimizer.init(v, np.zeros(v.shape))
    v = optimizer.run(iter=it, return_value='v', verbose=False)
    v_arr += [v]

# fidelity = FourMagMSE(y, 25**2 / (sigma**2), mask)
# # denoiser = dnsr.DnCNN('Denoisers/dnsr/DnCNN/weights/dncnn25_17.pth')
# denoiser.set_param(25)
# optimizer = optim.PPR(fidelity, denoiser)
# optimizer.init(v1, np.zeros(y.shape))
# v2 = optimizer.run(mask, (n, m), iter=pnpiter2, return_value='v', verbose=False)
#
# fidelity = FourMagMSE(y, 10**2 / (sigma**2), mask)
# # denoiser = dnsr.DnCNN('Denoisers/dnsr/DnCNN/weights/dncnn10_17.pth')
# denoiser.set_param(10)
# optimizer = optim.PPR(fidelity, denoiser)
# optimizer.init(v2, np.zeros(y.shape))
# v3 = optimizer.run(mask, (n, m), iter=pnpiter3, return_value='v', verbose=False)

for i in range(len(v_arr)):
    v_arr[i] = v_arr[i][mask].reshape(img.shape)
# v0 = v0[mask].reshape((n, m))
# v1 = v1[mask].reshape((n, m))
# v2 = v2[mask].reshape((n, m))
# v3 = v3[mask].reshape((n, m))

if args.display:
    tools.stackview([img, *v_arr], width=20, method='Pillow')

if args.save != None:
    res = np.clip(v_arr[-1] * 255, 0, 255)
    res = Image.fromarray(res.astype(np.uint8))
    res.save(args.save, format='PNG')
