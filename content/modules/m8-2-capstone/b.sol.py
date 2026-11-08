import math


def ema_update(ema, params, decay):
    return [decay * e + (1 - decay) * p for e, p in zip(ema, params)]


def timestep_embedding(t, d):
    h = d // 2
    w = [10000 ** (-i / h) for i in range(h)]
    return [math.sin(t * x) for x in w] + [math.cos(t * x) for x in w]


def ddim_timesteps(T, steps):
    if steps == 1:
        return [T - 1]
    return [int(round((T - 1) * (1 - k / (steps - 1)))) for k in range(steps)]


def nfe(steps, cfg, batched):
    return steps * (2 if cfg and not batched else 1)


def dice(pred, gt):
    a, b = sum(pred), sum(gt)
    if a + b == 0:
        return 1.0
    return 2 * sum(p and g for p, g in zip(pred, gt)) / (a + b)
