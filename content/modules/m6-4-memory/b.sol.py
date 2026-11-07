def checkpoint_cost(sizes, k):
    n = len(sizes)
    segs = [list(range(s, min(s + k, n))) for s in range(0, n, k)]
    peak = 0
    for j, seg in enumerate(segs):
        ends = sum(sizes[sg[-1]] for sg in segs[: j + 1])
        inner = sum(sizes[i] for i in seg[:-1])
        peak = max(peak, ends + inner)
    return peak, sum(len(sg) - 1 for sg in segs)


def best_k(sizes):
    best = None
    for k in range(1, len(sizes) + 1):
        p = checkpoint_cost(sizes, k)[0]
        if best is None or p < best[0]:
            best = (p, k)
    return best[1]


print(checkpoint_cost([1] * 16, 4))   # (7, 12)
print(checkpoint_cost([1] * 16, 1))   # (16, 0)
print(best_k([1] * 16))               # 3 (k = 3 ve 4 tepe 7 ile berabere; 3 daha az yeniden hesap)
