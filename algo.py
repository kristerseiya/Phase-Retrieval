
import numpy as np

import tools

def hio(y, mask, iter, beta=0.8):

    phase = np.random.rand(*y.shape) * 2 * np.pi
    prev = np.real(tools.ifft2d(y * np.exp(1j * phase)))
    for i in range(iter):

        print('Iteration #{:d} '.format(i+1), end='')

        pred = y * np.exp(1j * phase)
        pred = np.real(tools.ifft2d(pred))
        idx = (pred < 0) + (mask == False)
        pred[idx] = prev[idx] - beta * pred[idx]
        new_f = tools.fft2d(pred)
        phase = np.angle(new_f)
        prev = pred.copy()

        residual = np.power(y - np.abs(new_f), 2).mean()
        print(residual)

    return pred
