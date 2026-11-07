import math


def rnn_forward(xs, h0, Wx, Wh, b):
    hs, h = [], h0
    for x in xs:
        h = [math.tanh(sum(Wx[i][j] * x[j] for j in range(len(x))) + sum(Wh[i][k] * h[k] for k in range(len(h))) + b[i])
             for i in range(len(b))]
        hs.append(h)
    return hs


def rnn_backward(xs, hs, h0, Wx, Wh, dhs):
    """→ (dWx, dWh, db, dh0)"""
    H, D = len(Wx), len(Wx[0])
    dWx = [[0.0] * D for _ in range(H)]
    dWh = [[0.0] * H for _ in range(H)]
    db = [0.0] * H
    carry = [0.0] * H                                   # gelecekten gelen dolaylı gradyan
    for t in reversed(range(len(xs))):
        dh = [a + c for a, c in zip(dhs[t], carry)]
        da = [g * (1 - v * v) for g, v in zip(dh, hs[t])]  # tanh içinden
        prev = hs[t - 1] if t > 0 else h0
        for i in range(H):
            db[i] += da[i]
            for j in range(D):
                dWx[i][j] += da[i] * xs[t][j]
            for k in range(H):
                dWh[i][k] += da[i] * prev[k]
        carry = [sum(Wh[i][k] * da[i] for i in range(H)) for k in range(H)]
    return dWx, dWh, db, carry


hs = rnn_forward([[1.0], [0.0], [0.0]], [0.0], [[1.0]], [[0.5]], [0.0])
print(hs)                                                     # [[0.7616], [0.3634], [0.1797]]
print(rnn_backward([[1.0], [0.0], [0.0]], hs, [0.0], [[1.0]], [[0.5]], [[0.0], [0.0], [1.0]]))
