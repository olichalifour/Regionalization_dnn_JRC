import numpy as np


def ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a