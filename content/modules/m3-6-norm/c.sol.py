import numpy as np


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps, m = bn_param.get("eps", 1e-5), bn_param.get("momentum", 0.9)
    D = x.shape[1]
    rm = bn_param.setdefault("running_mean", np.zeros(D))
    rv = bn_param.setdefault("running_var", np.ones(D))
    if mode == "train":
        mu, var = x.mean(0), x.var(0)
        bn_param["running_mean"] = m * rm + (1 - m) * mu
        bn_param["running_var"] = m * rv + (1 - m) * var
    else:
        mu, var = rm, rv
    std = np.sqrt(var + eps)
    xh = (x - mu) / std
    return gamma * xh + beta, (xh, std, gamma)


def batchnorm_backward(dout, cache):
    xh, std, gamma = cache
    N = dout.shape[0]
    dxh = dout * gamma
    dx = (N * dxh - dxh.sum(0) - xh * (dxh * xh).sum(0)) / (N * std)
    return dx, (dout * xh).sum(0), dout.sum(0)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(1, keepdims=True)
    std = np.sqrt(x.var(1, keepdims=True) + eps)
    xh = (x - mu) / std
    return gamma * xh + beta, (xh, std, gamma)


def layernorm_backward(dout, cache):
    xh, std, gamma = cache
    D = dout.shape[1]
    dxh = dout * gamma
    dx = (D * dxh - dxh.sum(1, keepdims=True) - xh * (dxh * xh).sum(1, keepdims=True)) / (D * std)
    return dx, (dout * xh).sum(0), dout.sum(0)
