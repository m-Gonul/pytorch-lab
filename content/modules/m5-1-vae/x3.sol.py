import numpy as np


def lerp(z0, z1, t):
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    return (1 - t) * z0 + t * z1


def slerp(z0, z1, t):
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    om = np.arccos(np.clip(z0 @ z1 / np.linalg.norm(z0) / np.linalg.norm(z1), -1, 1))
    return (np.sin((1 - t) * om) * z0 + np.sin(t * om) * z1) / np.sin(om)


def typicality(Z):
    return np.linalg.norm(Z, axis=-1) / np.sqrt(Z.shape[-1])
