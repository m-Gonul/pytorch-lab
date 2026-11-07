def count(layers, in_shape):
    ...


def inception_params(cin, c1, c3r, c3, c5r, c5, pp):
    ...


def lrn(a, k=2, n=5, alpha=1e-4, beta=0.75):
    ...


cnn2 = [{"type": "conv", "k": 7, "cin": 1, "cout": 16, "groups": 1, "bias": True}, {"type": "pool", "k": 2}]
print(count(cnn2, (1, 28, 28)))
