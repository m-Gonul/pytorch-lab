def mergesort_dag(n):
    dur, preds = {}, {}

    def build(lo, hi):
        if hi - lo == 1:
            dur[("sort", lo, hi)] = 1
            preds[("sort", lo, hi)] = []
            return ("sort", lo, hi)
        mid = (lo + hi) // 2
        a, b = build(lo, mid), build(mid, hi)
        m = ("merge", lo, hi)
        dur[m], preds[m] = hi - lo, [a, b]
        s = ("sort", lo, hi)
        dur[s], preds[s] = 0, [m]
        return s

    build(0, n)
    return dur, preds


def max_concurrency(dur, preds):
    start, end = {}, {}

    def ef(v):
        if v not in end:
            start[v] = max((ef(u) for u in preds[v]), default=0)
            end[v] = start[v] + dur[v]
        return end[v]

    for v in preds:
        ef(v)
    events = sorted({t for v in preds for t in (start[v], end[v])})
    best = 0
    for t in events:
        best = max(best, sum(1 for v in preds if dur[v] > 0 and start[v] <= t < end[v]))
    return best


d, p = mergesort_dag(8)
print(len(d), max_concurrency(d, p))
