def grid_stride_indices(n, grid, block):
    T = grid * block
    return [list(range(t, n, T)) for t in range(T)]


def chunk_indices(n, grid, block):
    T = grid * block
    c = (n + T - 1) // T
    return [list(range(t * c, min(n, (t + 1) * c))) for t in range(T)]


def coalesced_fraction(assign, n_iter):
    ok = tot = 0
    for w in range(0, len(assign), 32):
        warp = assign[w:w + 32]
        for it in range(n_iter):
            idx = [a[it] for a in warp if it < len(a)]
            if len(idx) < 2:
                continue
            tot += 1
            ok += all(b - a == 1 for a, b in zip(idx, idx[1:]))
    return ok / tot if tot else 1.0


print(grid_stride_indices(10, 1, 4))
