import math


def positional_encoding(n, d):
    P = [[0.0] * d for _ in range(n)]
    for p in range(n):
        for i in range(0, d, 2):
            a = p / 10000 ** (i / d)
            P[p][i] = math.sin(a)
            if i + 1 < d:
                P[p][i + 1] = math.cos(a)
    return P


def noam_lr(step, d, warmup, factor=1.0):
    return factor * d ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def smooth_targets(t, V, eps):
    return [(1 - eps) * (i == t) + eps / V for i in range(V)]


def smoothed_ce(logp, t, eps):
    return -sum(q * l for q, l in zip(smooth_targets(t, len(logp), eps), logp))


def transformer_params(d, h, dff, N, V):
    mha = 4 * (d * d + d)
    ffn = 2 * d * dff + dff + d
    enc = mha + ffn + 4 * d
    dec = 2 * mha + ffn + 6 * d
    return {"enc_layer": enc, "dec_layer": dec, "total": N * (enc + dec) + V * d}


def causal_allowed(T):
    return T * (T + 1) // 2
