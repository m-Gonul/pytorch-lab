import numpy as np


def quant_per_tensor(W):
    ...


def quant_per_channel(W):
    ...


def int8_matmul(xq, sx, wq, sw):
    ...


def rel_error(y, y_ref):
    ...


rng = np.random.default_rng(0)
W = rng.normal(size=(64, 128)); W[3] *= 100       # aykırı kanal
x = rng.normal(size=(16, 128))
xq, sx = quant_per_tensor(x)
for name, (wq, sw) in [("tensör", quant_per_tensor(W)), ("kanal", quant_per_channel(W))]:
    print(name, rel_error(int8_matmul(xq, sx, wq, sw), x @ W.T))
