
import numpy as np
from PIL import Image
import argparse

import algo
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--iter', type=int, default=2000)
args = parser.parse_args()

# def stackview(imgs, width=20):
#     h = imgs[0].shape[0]
#     sep = np.zeros([h, width], dtype=np.uint8)
#     view = image_reformat(imgs[0])
#     for img in imgs[1:]:
#         view = np.concatenate([view, sep, image_reformat(img)], axis=1)
#     view = Image.fromarray(view, 'L')
#     view.show()

img = Image.open(args.image).convert('L')
img = np.array(img) / 255.
# img = img + np.random.normal(size=img.shape) * 0.1

pad_len_1 = img.shape[0] // 2
pad_len_2 = img.shape[1] // 2
img = np.pad(img, ((pad_len_1, pad_len_1), (pad_len_2, pad_len_2)), 'constant', constant_values=((0, 0), (0, 0)))
mask = np.ones(img.shape, dtype=bool) * False
mask[pad_len_1:-pad_len_1, pad_len_2:-pad_len_2] = True

n, m = img.shape
# sd = 0.1 * np.sqrt(n*m)
# noise = np.random.normal(size=img.shape) * sd + 1j * np.random.normal(size=img.shape) * sd
noise = 0
y = np.abs(tools.fft2d(img) + noise)

x = algo.hio(y, mask, args.iter)

print(x.shape)

tools.stackview([img, x])
