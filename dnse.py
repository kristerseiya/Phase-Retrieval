
import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F
from scipy.signal import correlate2d
from scipy.signal import fftconvolve

try:
    from prox_tv import tv1_2d as prx_tv
    import pybm3d
except:
    pass

from DnCNN.utils import load_dncnn
import tools


class ProxTV:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, x):
        return prx_tv(x, self.lambd)

class DnCNN:
    def __init__(self, model_path, use_tensor=False,
                 patch_size=-1, device=None, mask=None, imgshape=None):

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.net = load_dncnn(model_path, device=device)
        self.patch_size = patch_size
        self.use_tensor = use_tensor

    def __call__(self, x):
        if not self.use_tensor:
            x = torch.tensor(x, dtype=torch.float32,
                             requires_grad=False, device=self.device)
            if self.patch_size > 0:
                x = x.view(batch_size, 1, self.patch_size, self.patch_size)
            else:
                x = x.view(1, 1, *x.size())
            y = self.net(x)
            y = y.cpu().squeeze(0).squeeze(0)
            y = y.numpy()
            return y
        return self.net(x)

class BM3D:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return pybm3d.bm3d.bm3d(x, self.sigma)
