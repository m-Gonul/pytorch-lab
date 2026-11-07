import math


def sig(v):
    return 1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))


def affine(W, v, b):
    return [sum(w * x for w, x in zip(row, v)) + bi for row, bi in zip(W, b)]


def lstm_step(x, h, c, W_ih, W_hh, b_ih, b_hh):
    H = len(h)
    a = [p + q for p, q in zip(affine(W_ih, x, b_ih), affine(W_hh, h, b_hh))]
    i = [sig(v) for v in a[0:H]]
    f = [sig(v) for v in a[H:2 * H]]
    g = [math.tanh(v) for v in a[2 * H:3 * H]]
    o = [sig(v) for v in a[3 * H:4 * H]]
    c_new = [fk * ck + ik * gk for fk, ck, ik, gk in zip(f, c, i, g)]
    h_new = [ok * math.tanh(ck) for ok, ck in zip(o, c_new)]
    return h_new, c_new


def gru_step(x, h, W_ih, W_hh, b_ih, b_hh):
    H = len(h)
    gi, gh = affine(W_ih, x, b_ih), affine(W_hh, h, b_hh)
    r = [sig(gi[k] + gh[k]) for k in range(H)]
    z = [sig(gi[H + k] + gh[H + k]) for k in range(H)]
    n = [math.tanh(gi[2 * H + k] + r[k] * gh[2 * H + k]) for k in range(H)]
    return [(1 - z[k]) * n[k] + z[k] * h[k] for k in range(H)]
