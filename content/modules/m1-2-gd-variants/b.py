import random


def minibatches(n, batch_size, rng):
    ...


def sgd_epoch(xs, ys, w, b, lr, batch_size, rng):
    ...


def momentum_path(grad, w0, lr, beta, steps):
    ...


print(minibatches(10, 4, random.Random(0)))
print(momentum_path(lambda w: 2 * (w - 3), 0.0, 0.25, 0.5, 4))   # [1.5, 3.0, …]
