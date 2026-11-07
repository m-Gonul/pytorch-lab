def ring_allreduce(vecs):
    ...


res, sent = ring_allreduce([[1, 3, 0, 0], [3, 5, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
print(res[0], sent)    # [4, 8, 2, 2], her işçi 2·3·1 = 6 eleman
