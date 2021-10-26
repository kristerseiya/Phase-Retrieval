
import numpy as np
from PIL import Image
from scipy.fftpack import fft, ifft, fftshift
import argparse

import dnsr
import optim
import algo
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--hioiter', type=int, default=50)
parser.add_argument('--pnpiter', type=int, default=600)
parser.add_argument('--alpha', type=float, default=2500)
parser.add_argument('--lambd', type=float, default=1)
parser.add_argument('--noise', type=float, default=1)
parser.add_argument('--samprate', type=float, default=4)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--display', action='store_true')
parser.add_argument('--save', type=str, default=None)
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
        # return self.alpha * np.real(np.conjugate(tools.fft2d(np.conjugate(f) - self.y * np.conjugate(f) / np.abs(f))))

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

y = np.real(np.abs(tools.fft2d(img))) + np.random.normal(size=img.shape) * args.noise / 255.
# y = np.real(np.abs(tools.fft2d(img)))
# alpha = 10 / 255.
# intensity_noise = alpha * y * np.random.normal(size=img.shape)
# y = (y**2 + intensity_noise)
# y = y * (y > 0)
# y = np.sqrt(y)

v0 = algo.hio(y, mask, args.hioiter, beta=args.beta, verbose=False)

v0[~mask] = np.zeros(img.size - mask.sum())

if args.noise > 0:
    sigma = args.noise
else:
    sigma = 1

fidelity = FourMagMSE(y, 50**2 / (sigma**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn50_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v0, np.zeros(y.shape))
v1 = optimizer.run(mask, (n, m), iter=pnpiter1, return_value='v', verbose=False, beta=args.beta)

fidelity = FourMagMSE(y, 25**2 / (sigma**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn25_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v1, np.zeros(y.shape))
v2 = optimizer.run(mask, (n, m), iter=pnpiter2, return_value='v', verbose=False, beta=args.beta)

fidelity = FourMagMSE(y, 10**2 / (sigma**2), mask)
denoiser = dnsr.DnCNN('DnCNN/weights/dncnn10_17.pth')
optimizer = optim.PnPADMMHIO(fidelity, denoiser)
optimizer.init(v2, np.zeros(y.shape))
v3 = optimizer.run(mask, (n, m), iter=pnpiter3, return_value='v', verbose=False, beta=args.beta)

img = img[mask].reshape((n, m))
v0 = v0[mask].reshape((n, m))
v1 = v1[mask].reshape((n, m))
v2 = v2[mask].reshape((n, m))
v3 = v3[mask].reshape((n, m))

if args.display:
    tools.stackview([img, v0, v1, v2, v3], width=20, method='Pillow')

if args.save != None:
    res = np.clip(v3 * 255, 0, 255)
    res = Image.fromarray(res.astype(np.uint8))
    res.save(args.save, format='PNG')
