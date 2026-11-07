import math
from collections import Counter, defaultdict


def fit_nb(rows, labels, alpha=1.0):
    n, d = len(rows), len(rows[0])
    class_count = Counter(labels)
    counts = [defaultdict(Counter) for _ in range(d)]       # counts[j][c][v]
    values = [set() for _ in range(d)]
    for row, c in zip(rows, labels):
        for j, v in enumerate(row):
            counts[j][c][v] += 1
            values[j].add(v)
    return {"n": n, "class_count": class_count, "counts": counts, "values": values, "alpha": alpha}


def predict_proba(model, row):
    a, logs = model["alpha"], {}
    for c, nc in model["class_count"].items():
        s = math.log(nc / model["n"])
        for j, v in enumerate(row):
            p = (model["counts"][j][c][v] + a) / (nc + a * len(model["values"][j]))
            if p == 0:
                s = -math.inf
                break
            s += math.log(p)
        logs[c] = s
    m = max(logs.values())
    if m == -math.inf:
        return {c: 1 / len(logs) for c in logs}
    ex = {c: math.exp(s - m) for c, s in logs.items()}
    z = sum(ex.values())
    return {c: e / z for c, e in ex.items()}


def predict(model, row):
    p = predict_proba(model, row)
    return max(p, key=p.get)


rows = [("g", "z"), ("g", "s"), ("y", "s"), ("y", "z"), ("g", "s"), ("y", "s")]
labels = ["E", "E", "H", "E", "H", "H"]
print(predict_proba(fit_nb(rows, labels, alpha=0.0), ("g", "s")))   # E: 0.4
print(predict_proba(fit_nb(rows, labels, alpha=1.0), ("g", "s")))   # E: 0.4286
