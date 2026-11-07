import numpy as np


def warp_sectors(addr):
    s = np.sort(np.asarray(addr) // 32, axis=1)
    return 1 + (np.diff(s, axis=1) != 0).sum(1)


def warp_bank_degree(words):
    w = np.sort(np.asarray(words), axis=1)
    first = np.concatenate([np.ones((w.shape[0], 1), bool), np.diff(w, axis=1) != 0], 1)
    rows = np.broadcast_to(np.arange(w.shape[0])[:, None], w.shape)
    cnt = np.zeros((w.shape[0], 32), np.int64)
    np.add.at(cnt, (rows[first], (w % 32)[first]), 1)
    return cnt.max(1)


def _warps(a):
    return a.reshape(-1, 32)


def transpose_traffic(N, kind, T=32, pad=0):
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if kind in ("copy", "naive"):
        rd = _warps(4 * (y * N + x))
        wr = _warps(4 * (y * N + x)) if kind == "copy" else _warps(4 * (x * N + y))
        return int(warp_sectors(rd).sum()), int(warp_sectors(wr).sum()), 0.0
    nb = N // T
    by, bx, ty, tx = np.meshgrid(np.arange(nb), np.arange(nb), np.arange(T), np.arange(T), indexing="ij")
    rd = _warps(4 * ((by * T + ty) * N + bx * T + tx))
    wr = _warps(4 * ((bx * T + ty) * N + by * T + tx))
    sh = _warps(tx * (T + pad) + ty)
    return int(warp_sectors(rd).sum()), int(warp_sectors(wr).sum()), float(warp_bank_degree(sh).mean())


print(transpose_traffic(256, "naive"), transpose_traffic(256, "tiled", pad=1))
