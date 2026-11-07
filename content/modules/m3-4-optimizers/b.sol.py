import math


def sgd_momentum(w, g, state, lr, beta=0.9):
    u = state.get("u")
    u = list(g) if u is None else [beta * a + b for a, b in zip(u, g)]
    state["u"] = u
    return [wi - lr * ui for wi, ui in zip(w, u)]


def rmsprop(w, g, state, lr=0.01, alpha=0.99, eps=1e-8):
    s = state.get("s", [0.0] * len(w))
    s = [alpha * a + (1 - alpha) * b * b for a, b in zip(s, g)]
    state["s"] = s
    return [wi - lr * gi / (math.sqrt(si) + eps) for wi, gi, si in zip(w, g, s)]


def adam(w, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False):
    t = state.get("t", 0) + 1
    m = state.get("m", [0.0] * len(w))
    v = state.get("v", [0.0] * len(w))
    if decoupled:
        w = [wi * (1 - lr * wd) for wi in w]
    else:
        g = [gi + wd * wi for gi, wi in zip(g, w)]
    m = [b1 * a + (1 - b1) * b for a, b in zip(m, g)]
    v = [b2 * a + (1 - b2) * b * b for a, b in zip(v, g)]
    state.update(t=t, m=m, v=v)
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    return [wi - lr * (mi / bc1) / (math.sqrt(vi / bc2) + eps) for wi, mi, vi in zip(w, m, v)]


w, st = [0.0], {}
for _ in range(2):
    w = adam(w, [2 * (w[0] - 3)], st, lr=0.1, eps=0.0)
print(w)       # [0.19989…]
