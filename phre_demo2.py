
import numpy as np
import torch
from PIL import Image
from scipy.fftpack import fft, ifft, fftshift
import argparse

import optim
import algo
import tools
from Denoisers import dnsr
import likelihood

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
# imgpad = imgpad * np.exp(1j * imgpad * np.pi)

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
    y = yy + (np.random.normal(size=imgpad.shape) + 1j * np.random.normal(size=imgpad.shape)) * args.noiselvl / 255.
    y = np.abs(y)
    sigma = args.noiselvl if args.noiselvl > 0 else 1
    # sigma = np.std(y - np.abs(yy))*255

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

y_t = tools.numpy2torch(y).type(torch.float32)
v_t = tools.numpy2torch(v).type(torch.float32)
mask_t = tools.numpy2torch(mask)

class SpacePrior:
    def __init__(self, mask, supdim):
        self.mask = mask
        self.supdim = supdim
        self.denoiser = dnsr.cDnCNN('Denoisers/dnsr/DnCNN/weights/cDnCNNv6_1-50.pth', use_tensor=True)

    def prox(self, x):
        v = torch.zeros_like(x)
        xs = (x[self.mask]).reshape(self.supdim)
        xs = xs.unsqueeze(0).unsqueeze(0)
        v[self.mask] = self.denoiser(xs).flatten()

        idx = (v < 0) * self.mask
        v[idx] = torch.zeros(idx.sum())
        return v

prior = SpacePrior(mask_t, img.shape[::-1])

for it, a in zip(args.pnpiter, args.sigma):
    fidelity = likelihood.FourierMagRician(y_t, sigma/255., a/255.)
    prior.denoiser.set_param(a/255.)
    optimizer = optim.ADMM(fidelity, prior)
    optimizer.init(v_t, torch.zeros_like(v_t))
    v_t = optimizer.run(iter=it, return_value='v', verbose=True)
    v_arr += [tools.torch2numpy(v_t)]

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
