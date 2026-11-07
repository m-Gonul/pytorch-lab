import math


def arange(n):
    return list(range(n))


def load(mem, offs, mask, other=0.0):
    return [mem[o] if m else other for o, m in zip(offs, mask)]


def store(mem, offs, vals, mask):
    for o, v, m in zip(offs, vals, mask):
        if m:
            mem[o] = v


def launch(program, grid, *args):
    for pid in range(grid):
        program(pid, *args)


def add_program(pid, x, y, out, n, BLOCK):
    offs = [pid * BLOCK + i for i in arange(BLOCK)]
    mask = [o < n for o in offs]
    a = load(x, offs, mask)
    b = load(y, offs, mask)
    store(out, offs, [u + v for u, v in zip(a, b)], mask)


def softmax_program(pid, X, Y, n_cols, BLOCK):
    cols = arange(BLOCK)
    mask = [c < n_cols for c in cols]
    offs = [pid * n_cols + c for c in cols]
    row = load(X, offs, mask, other=-math.inf)
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    store(Y, offs, [v / s for v in e], mask)


n = 1000
x, y, out = list(range(n)), [1.0] * n, [None] * n
launch(add_program, (n + 255) // 256, x, y, out, n, 256)
print(out[:3], out[-1])
X = [-5.0, -6.0, -7.0, 1.0, 2.0, 3.0]; Y = [None] * 6
launch(softmax_program, 2, X, Y, 3, 4)
print(Y)
