def inclusive_scan(x, op):
    out, acc = [], None
    for i, v in enumerate(x):
        acc = v if i == 0 else op(acc, v)
        out.append(acc)
    return out


def exclusive_scan(x, op, identity):
    out, acc = [], identity
    for v in x:
        out.append(acc)
        acc = op(acc, v)
    return out


def kogge_stone(x, op):
    buf, n, s, steps, ops = list(x), len(x), 1, 0, 0
    while s < n:
        new = buf[:]
        for i in range(s, n):
            new[i] = op(buf[i - s], buf[i])
            ops += 1
        buf, s, steps = new, s * 2, steps + 1
    return buf, steps, ops


def kogge_stone_inplace(x, order):
    a, n, s = list(x), len(x), 1
    while s < n:
        idx = range(s, n) if order == "asc" else range(n - 1, s - 1, -1)
        for i in idx:
            a[i] = a[i - s] + a[i]
        s *= 2
    return a


def brent_kung(x, op):
    a, n = list(x), len(x)
    steps = ops = 0
    s = 1
    while s < n:
        for i in range(2 * s - 1, n, 2 * s):
            a[i] = op(a[i - s], a[i])
            ops += 1
        steps += 1
        s *= 2
    s = n // 4
    while s >= 1:
        for j in range(2 * s - 1, n - s, 2 * s):
            a[j + s] = op(a[j], a[j + s])
            ops += 1
        steps += 1
        s //= 2
    return a, steps, ops


def hierarchical_scan(x, section, op):
    parts = [inclusive_scan(x[i:i + section], op) for i in range(0, len(x), section)]
    totals = inclusive_scan([p[-1] for p in parts], op)
    out = list(parts[0])
    for k in range(1, len(parts)):
        out += [op(totals[k - 1], v) for v in parts[k]]
    return out


add = lambda a, b: a + b
print(kogge_stone([3, 1, 7, 0, 4, 1, 6, 3], add), brent_kung([3, 1, 7, 0, 4, 1, 6, 3], add))
