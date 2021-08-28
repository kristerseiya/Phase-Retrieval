
import numpy as np
from PIL import Image
from scipy.fftpack import fft, ifft
import argparse

import dnsr
import optim
import algo
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--iter', type=int, nargs='*', default=[50, 200, 200, 200, 200])
parser.add_argument('--alpha', type=float, default=2500)
parser.add_argument('--lambd', type=float, default=1)
parser.add_argument('--noise', type=float, default=1)
parser.add_argument('--samprate', type=float, default=2)
parser.add_argument('--beta', type=float, default=0.9)
# parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

iters = [50, 200, 200, 200, 200]

if len(args.iter) > 5:
    raise ValueError('Too many --iter')

for i, it in enumerate(args.iter):
    iters[i] = it

tau = 1

class DiffractFidelity:
    def __init__(self, y, alpha, mask):
        self.y = y
        self.alpha = alpha
        self.mask = mask

    def __call__(self, x):
        f = tools.fft2d(x)
        return np.real(self.alpha * np.sum((y - np.abs(f))**2))

    def grad(self, x):
        f = tools.fft2d(x)
        return self.alpha * np.conjugate(tools.fft2d(np.conjugate(f) - self.y * np.conjugate(f) / np.abs(f)))

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

# y = np.real(np.abs(tools.fft2d(img))) + np.random.normal(size=img.shape) * args.noise / 255.
y = np.real(np.abs(tools.fft2d(img)))
alpha = 10 / 255.
intensity_noise = alpha * y * np.random.normal(size=img.shape)
y = (y**2 + intensity_noise)
y = y * (y > 0)
y = np.sqrt(y)

v0 = algo.hio(y, mask, iters[0], beta=args.beta)

v0[~mask] = np.zeros(img.size - mask.sum())

fidelity = DiffractFidelity(y, 65**2 / (args.noise**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn65_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v0, np.zeros(y.shape))
v1 = optimizer.run(mask, (n, m), iter=iters[1], return_value='v', verbose=True, beta=args.beta)

fidelity = DiffractFidelity(y, 50**2 / (args.noise**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn50_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v1, np.zeros(y.shape))
v2 = optimizer.run(mask, (n, m), iter=iters[2], return_value='v', verbose=True, beta=args.beta)

fidelity = DiffractFidelity(y, 25**2 / (args.noise**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn25_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v2, np.zeros(y.shape))
v3 = optimizer.run(mask, (n, m), iter=iters[3], return_value='v', verbose=True, beta=args.beta)

fidelity = DiffractFidelity(y, 10**2 / (args.noise**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn10_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v3, np.zeros(y.shape))
v4 = optimizer.run(mask, (n, m), iter=iters[4], return_value='v', verbose=True, beta=args.beta)

img = img[mask].reshape((n, m))
v0 = v0[mask].reshape((n, m))
v1 = v1[mask].reshape((n, m))
v2 = v2[mask].reshape((n, m))
v3 = v3[mask].reshape((n, m))
v4 = v4[mask].reshape((n, m))

tools.stackview([img, v0, v1, v2, v3, v4], width=20, method='Pillow')
