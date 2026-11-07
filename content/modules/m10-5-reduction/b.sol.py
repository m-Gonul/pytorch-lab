def reduce_tree(xs, op, identity):
    xs, steps, ops = list(xs), 0, 0
    while len(xs) > 1:
        nxt = []
        for i in range(0, len(xs), 2):
            if i + 1 < len(xs):
                nxt.append(op(xs[i], xs[i + 1]))
                ops += 1
            else:
                nxt.append(xs[i])
        xs = nxt
        steps += 1
    return (xs[0] if xs else identity), steps, ops


def _strides(n, mapping):
    out, s = [], (1 if mapping == "simple" else n)
    while 1 <= s <= n:
        out.append(s)
        s = s * 2 if mapping == "simple" else s // 2
    return out


def _active(t, s, mapping):
    return t % s == 0 if mapping == "simple" else t < s


def warp_activity(n_threads, mapping):
    res = []
    for s in _strides(n_threads, mapping):
        act = [t for t in range(n_threads) if _active(t, s, mapping)]
        warps = {}
        for t in range(n_threads):
            warps.setdefault(t // 32, []).append(_active(t, s, mapping))
        aw = sum(any(v) for v in warps.values())
        dw = sum(any(v) and not all(v) for v in warps.values())
        res.append((len(act), aw, dw))
    return res


def run_block(x, mapping):
    a = list(x)
    n = len(a) // 2
    for s in _strides(n, mapping):
        for t in range(n):
            if _active(t, s, mapping):
                i = 2 * t if mapping == "simple" else t
                a[i] += a[i + s]
    return a[0]


def hist_private(data, n_threads, n_bins):
    priv = [[0] * n_bins for _ in range(n_threads)]
    for i, v in enumerate(data):
        priv[i % n_threads][v] += 1
    out = [0] * n_bins
    merges = 0
    for p in priv:
        for b in range(n_bins):
            out[b] += p[b]
            merges += 1
    return out, merges


print(reduce_tree([3, 1, 7, 0, 4, 1, 6, 3], max, float("-inf")))
