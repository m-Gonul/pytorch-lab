def ring_allreduce(vecs):
    N = len(vecs)
    L = len(vecs[0])
    c = L // N
    buf = [list(v) for v in vecs]
    sent = [0] * N
    sl = lambda k: range(k * c, (k + 1) * c)
    for s in range(N - 1):
        msgs = [(i, (i - s) % N, [buf[i][j] for j in sl((i - s) % N)]) for i in range(N)]
        for i, k, data in msgs:
            dst = (i + 1) % N
            for j, val in zip(sl(k), data):
                buf[dst][j] += val
            sent[i] += len(data)
    for s in range(N - 1):
        msgs = [(i, (i + 1 - s) % N, [buf[i][j] for j in sl((i + 1 - s) % N)]) for i in range(N)]
        for i, k, data in msgs:
            dst = (i + 1) % N
            for j, val in zip(sl(k), data):
                buf[dst][j] = val
            sent[i] += len(data)
    return buf, sent


res, sent = ring_allreduce([[1, 3, 0, 0], [3, 5, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
print(res[0], sent)    # [4, 8, 2, 2], her işçi 2·3·1 = 6 eleman
