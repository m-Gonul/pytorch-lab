import heapq


def amdahl(f, n):
    return 1 / ((1 - f) + f / n)


def gustafson(f, n):
    return (1 - f) + f * n


def karp_flatt(S, p):
    return (1 / S - 1 / p) / (1 - 1 / p)


def pipeline_makespan(stages, n_items):
    prev = [0.0] * len(stages)
    for _ in range(n_items):
        cur, t = [], 0.0
        for s, d in enumerate(stages):
            t = max(t, prev[s]) + d
            cur.append(t)
        prev = cur
    return prev[-1]


def _topo(preds):
    order, seen = [], set()

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        for u in preds[v]:
            visit(u)
        order.append(v)

    for v in preds:
        visit(v)
    return order


def work_span(dur, preds):
    ef = {}
    for v in _topo(preds):
        ef[v] = dur[v] + max((ef[u] for u in preds[v]), default=0)
    return sum(dur.values()), max(ef.values())


def list_schedule(dur, preds, p):
    succ = {v: [] for v in preds}
    for v, ps in preds.items():
        for u in ps:
            succ[u].append(v)
    bl = {}
    for v in reversed(_topo(preds)):
        bl[v] = dur[v] + max((bl[w] for w in succ[v]), default=0)
    remaining = {v: len(ps) for v, ps in preds.items()}
    ready = [(-bl[v], v) for v, k in remaining.items() if k == 0]
    heapq.heapify(ready)
    running, t, free, done = [], 0.0, p, 0
    while done < len(preds):
        while free and ready:
            _, v = heapq.heappop(ready)
            heapq.heappush(running, (t + dur[v], v))
            free -= 1
        t, v = heapq.heappop(running)
        free += 1
        done += 1
        for w in succ[v]:
            remaining[w] -= 1
            if remaining[w] == 0:
                heapq.heappush(ready, (-bl[w], w))
    return t


dur = {"a": 2, "b": 3, "c": 1, "d": 4}
preds = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
print(work_span(dur, preds), list_schedule(dur, preds, 2))
