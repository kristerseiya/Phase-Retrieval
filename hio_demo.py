
import numpy as np
from PIL import Image
import argparse

import algo
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--samprate', type=float, default=2)
parser.add_argument('--iter', type=int, default=2000)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--noise', type=float, default=0.)
parser.add_argument('--display', action='store_true')
parser.add_argument('--save', type=str, default=None)
args = parser.parse_args()

img = Image.open(args.image).convert('L')
img = np.array(img) / 255.
n, m = img.shape

pad_len_1 = int(img.shape[0] * (np.sqrt(args.samprate) - 1)) // 2
pad_len_2 = int(img.shape[1] * (np.sqrt(args.samprate) - 1)) // 2
n, m = img.shape
img = np.pad(img, ((pad_len_1, pad_len_1), (pad_len_2, pad_len_2)), 'constant', constant_values=((0, 0), (0, 0)))
mask = np.ones(img.shape, dtype=bool) * False
mask[pad_len_1:-pad_len_1, pad_len_2:-pad_len_2] = True

# sd = 0.1 * np.sqrt(n*m)
# noise = np.random.normal(size=img.shape) * sd + 1j * np.random.normal(size=img.shape) * sd
# noise = 0
y = np.abs(tools.fft2d(img)) + np.random.normal(size=img.shape) * args.noise / 255.

x = algo.hio(y, mask, args.iter, beta=args.beta)
# x2 = algo.oss(y, mask)

img = img[mask].reshape((n, m))
x = x[mask].reshape((n, m))
# x2 = x2[mask].reshape((n, m))


if args.display:
    tools.stackview([img, x], method='Pillow')

if args.save != None:
    res = np.clip(x * 255, 0, 255)
    res = Image.fromarray(res.astype(np.uint8))
    res.save(args.save, format='PNG')
