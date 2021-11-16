
import numpy as np
from PIL import Image
import argparse

import algo
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--samprate', type=float, default=4)
parser.add_argument('--iter', type=int, default=2000)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--noise', help='noise type (\'gaussian\' or \'poisson\')', type=str, default='gaussian')
parser.add_argument('--noiselvl', help='standard deviation if noise type is gaussian,'
                            'if poisson, the product of this parameter and the signal'
                            'would be used as a standard deviation of gaussian to'
                            'simulate a poisson noise', type=float, default=1)
parser.add_argument('--display', action='store_true')
parser.add_argument('--save', type=str, default=None)
args = parser.parse_args()

img = Image.open(args.image).convert('L')
img = np.array(img) / 255.
n, m = img.shape

pad_len_1 = int(img.shape[0] * (np.sqrt(args.samprate) - 1)) // 2
pad_len_2 = int(img.shape[1] * (np.sqrt(args.samprate) - 1)) // 2
imgpad = np.pad(img, ((pad_len_1, pad_len_1), (pad_len_2, pad_len_2)), 'constant', constant_values=((0, 0), (0, 0)))
mask = np.ones(imgpad.shape, dtype=bool) * False
mask[pad_len_1:-pad_len_1, pad_len_2:-pad_len_2] = True

if args.noise == 'gaussian':
    y = np.real(np.abs(tools.fft2d(imgpad))) + np.random.normal(size=imgpad.shape) * args.noiselvl / 255.

elif args.noise == 'poisson':
    yy = np.real(np.abs(tools.fft2d(imgpad)))
    alpha = args.noiselvl / 255.
    intensity_noise = alpha * yy * np.random.normal(size=imgpad.shape)
    y = (yy**2 + intensity_noise)
    y = y * (y > 0)
    y = np.sqrt(y)

elif args.noise == 'rician':
    yy = tools.fft2d(img)
    y = yy + (np.random.normal(size=imgpad.shape) + 1j * np.random.normal(size=imgpad.shape)) * args.noiselvl / 255. / np.sqrt(2)
    y = np.abs(y)

x = algo.hio(y, mask, args.iter, beta=args.beta)
# x = algo.oss(y, mask)
# x = algo.wf(y, mask, args.iter)

x = x[mask].reshape(img.shape)

if args.display:
    tools.stackview([img, x], method='Pillow')

if args.save != None:
    res = np.clip(x * 255, 0, 255)
    res = Image.fromarray(res.astype(np.uint8))
    res.save(args.save, format='PNG')
