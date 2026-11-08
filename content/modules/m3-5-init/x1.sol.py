import numpy as np

LAM, ALPHA = 1.0507009873554805, 1.6732632423543772


def selu(z):
    return LAM * np.where(z > 0, z, ALPHA * (np.exp(np.minimum(z, 0)) - 1))


def gauss_moments(f, n=20001):
    z = np.linspace(-10, 10, n)
    p = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    y = f(z)
    dz = z[1] - z[0]
    trap = lambda v: float(dz * (v.sum() - 0.5 * (v[0] + v[-1])))
    return trap(y * p), trap(y * y * p)


def deep_stats(act, init, depth, width, seed):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((1000, width))
    var = {"lecun": 1.0, "he": 2.0, "glorot": 1.0}[init] / width
    out = []
    for _ in range(depth):
        H = act(H @ (rng.standard_normal((width, width)) * np.sqrt(var)))
        out.append((float(H.mean()), float(H.std())))
    return out
