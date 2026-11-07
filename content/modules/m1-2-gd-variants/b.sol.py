import random


def minibatches(n, batch_size, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def sgd_epoch(xs, ys, w, b, lr, batch_size, rng):
    for batch in minibatches(len(xs), batch_size, rng):
        gw = gb = 0.0
        for i in batch:
            e = w * xs[i] + b - ys[i]
            gw += 2 * e * xs[i] / len(batch)
            gb += 2 * e / len(batch)
        w, b = w - lr * gw, b - lr * gb
    return w, b


def momentum_path(grad, w0, lr, beta, steps):
    w, v, path = w0, 0.0, []
    for _ in range(steps):
        v = beta * v + grad(w)
        w = w - lr * v
        path.append(w)
    return path


print(minibatches(10, 4, random.Random(0)))
print(momentum_path(lambda w: 2 * (w - 3), 0.0, 0.25, 0.5, 4))   # [1.5, 3.0, …]
