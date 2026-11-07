import numpy as np


def make_noise(kind, shape, rng, **p):
    if kind == "gaussian":
        return rng.normal(p["mean"], p["sigma"], shape)
    if kind == "rayleigh":
        return p["a"] + np.sqrt(-p["b"] * np.log(1 - rng.random(shape)))
    if kind == "erlang":
        return rng.gamma(p["b"], 1 / p["a"], shape)
    if kind == "exponential":
        return rng.exponential(1 / p["a"], shape)
    if kind == "uniform":
        return rng.uniform(p["a"], p["b"], shape)
    raise ValueError(kind)


def estimate(kind, samples):
    m, v = samples.mean(), samples.var()
    if kind == "gaussian":
        return {"mean": m, "sigma": np.sqrt(v)}
    if kind == "rayleigh":
        b = 4 * v / (4 - np.pi)
        return {"a": m - np.sqrt(np.pi * b / 4), "b": b}
    if kind == "erlang":
        return {"a": m / v, "b": m * m / v}
    if kind == "exponential":
        return {"a": 1 / m}
    if kind == "uniform":
        h = np.sqrt(3 * v)
        return {"a": m - h, "b": m + h}
    raise ValueError(kind)


def add_salt_pepper(img, ps, pp, rng):
    u = rng.random(img.shape)
    out = img.copy()
    out[u < pp] = 0
    out[(u >= pp) & (u < pp + ps)] = 255
    return out


rng = np.random.default_rng(0)
z = make_noise("rayleigh", (100000,), rng, a=10, b=400)
print(estimate("rayleigh", z))
