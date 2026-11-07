import numpy as np


def quant_per_tensor(W):
    s = np.abs(W).max() / 127
    return np.clip(np.round(W / s), -127, 127).astype(np.int8), s


def quant_per_channel(W):
    s = np.abs(W).max(1) / 127
    return np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8), s


def int8_matmul(xq, sx, wq, sw):
    return (xq.astype(np.int32) @ wq.T.astype(np.int32)) * sx * sw


def rel_error(y, y_ref):
    return np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)


rng = np.random.default_rng(0)
W = rng.normal(size=(64, 128)); W[3] *= 100       # aykırı kanal
x = rng.normal(size=(16, 128))
xq, sx = quant_per_tensor(x)
for name, (wq, sw) in [("tensör", quant_per_tensor(W)), ("kanal", quant_per_channel(W))]:
    print(name, rel_error(int8_matmul(xq, sx, wq, sw), x @ W.T))
