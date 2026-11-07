import math


def logsumexp(z):
    m = max(z)
    return m + math.log(sum(math.exp(v - m) for v in z))


def softmax(z):
    lse = logsumexp(z)
    return [math.exp(v - lse) for v in z]


def cross_entropy(z, y):
    return logsumexp(z) - z[y]


def grad_z(z, y):
    p = softmax(z)
    p[y] -= 1.0
    return p


print(softmax([2, 1, 0]))                  # [0.665, 0.245, 0.090]
print(cross_entropy([2, 1, 0], 0))         # 0.4076
print(softmax([1000, 999, 998]))           # aynı, taşma yok
