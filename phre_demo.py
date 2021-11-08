
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
parser.add_argument('--pnpiter', help='total number of iterations with PnP-ADMM', type=int, default=600)
parser.add_argument('--noise', help='noise type (\'gaussian\' or \'poisson\')', type=str, default='gaussian')
parser.add_argument('--noiselvl', help='standard deviation if noise type is gaussian,'
                            'if poisson, the product of this parameter and the signal'
                            'would be used as a standard deviation of gaussian to'
                            'simulate a poisson noise', type=float, default=1)
parser.add_argument('--samprate', help='oversampling rate for the measurement', type=float, default=4)
parser.add_argument('--beta', help='parameter for HIO', type=float, default=0.9)
parser.add_argument('--display', help='display result if given', action='store_true')
parser.add_argument('--save', help='save result if given a filename', type=str, default=None)
args = parser.parse_args()

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

    def prox(self, x, tau=1):
        xf = tools.fft2d(x)
        mag = 1 / (self.alpha + tau) * (self.alpha * self.y + tau * np.abs(xf))
        res = mag * np.exp(1j*np.angle(xf))
        res = np.real(tools.ifft2d(res))
        return res

img = Image.open(args.image).convert('L')
img = np.array(img) / 255

pad_len_1 = int(img.shape[0] * (np.sqrt(args.samprate) - 1)) // 2
pad_len_2 = int(img.shape[1] * (np.sqrt(args.samprate) - 1)) // 2
n, m = img.shape
img = np.pad(img, ((pad_len_1, pad_len_1), (pad_len_2, pad_len_2)), 'constant', constant_values=((0, 0), (0, 0)))
mask = np.ones(img.shape, dtype=bool) * False
mask[pad_len_1:-pad_len_1, pad_len_2:-pad_len_2] = True

if args.noise == 'gaussian':
    y = np.real(np.abs(tools.fft2d(img))) + np.random.normal(size=img.shape) * args.noiselvl / 255.
    # y = tools.fft2d(img) + (np.random.normal(size=img.shape) + 1j * np.random.normal(size=img.shape)) * args.noiselvl / 255. / np.sqrt(2)
    # y = np.abs(y)
    # print(np.std(y - np.abs(tools.fft2d(img)))*255)
    sigma = args.noiselvl if args.noiselvl > 0 else 1

elif args.noise == 'poisson':
    yy = np.real(np.abs(tools.fft2d(img)))
    alpha = args.noiselvl / 255.
    intensity_noise = alpha * yy * np.random.normal(size=img.shape)
    y = (yy**2 + intensity_noise)
    y = y * (y > 0)
    y = np.sqrt(y)
    sigma = np.std(y - yy) * 255.

v = algo.hio(y, mask, args.hioiter, beta=args.beta, verbose=False)

v[~mask] = np.zeros(img.size - mask.sum())

v_arr = []
v_arr += [v]

alpha = np.array([60., 50, 40, 30, 20, 10])
denoiser = dnsr.cDnCNN('Denoisers/dnsr/DnCNN/weights/cDnCNNv6_1-50.pth')
# denoiser = dnsr.BM3D()

for a in alpha:
    fidelity = FourMagMSE(y, a**2 / (sigma**2), mask)
    denoiser.set_param(a)
    denoiser.set_param(a / 255.)
    optimizer = optim.PPR(fidelity, denoiser)
    optimizer.init(v, np.zeros(y.shape))
    v = optimizer.run(mask, (n, m), iter=100, return_value='v', verbose=False)
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

img = img[mask].reshape((n, m))
for i in range(len(v_arr)):
    v_arr[i] = v_arr[i][mask].reshape((n, m))
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
