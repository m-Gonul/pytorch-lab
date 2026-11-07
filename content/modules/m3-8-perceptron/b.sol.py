import math


def tlu(w, b, x):
    return 1 if sum(wi * xi for wi, xi in zip(w, x)) + b >= 0 else 0


def train_perceptron(X, y, eta=1.0, max_epochs=100):
    w, b, errors = [0.0] * len(X[0]), 0.0, 0
    for _ in range(max_epochs):
        mistakes = 0
        for x, t in zip(X, y):
            d = t - tlu(w, b, x)
            if d:
                w = [wi + eta * d * xi for wi, xi in zip(w, x)]
                b += eta * d
                mistakes += 1
        errors += mistakes
        if mistakes == 0:
            return w, b, errors, True
    return w, b, errors, False


def margin(w, b, X, y):
    n = math.sqrt(sum(wi * wi for wi in w))
    return min((2 * t - 1) * (sum(wi * xi for wi, xi in zip(w, x)) + b) / n for x, t in zip(X, y))


def xor_net(x):
    h1 = tlu((1, 1), -0.5, x)
    h2 = tlu((1, 1), -1.5, x)
    return tlu((1, -1), -0.5, (h1, h2))


print([xor_net(x) for x in ((0, 0), (0, 1), (1, 0), (1, 1))])
