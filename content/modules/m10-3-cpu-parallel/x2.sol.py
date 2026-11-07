from collections import OrderedDict


def trace(order, n, T=4):
    if order == "ijk":
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    yield ("A", i * n + k)
                    yield ("B", k * n + j)
                yield ("C", i * n + j)
    elif order == "ikj":
        for i in range(n):
            for k in range(n):
                yield ("A", i * n + k)
                for j in range(n):
                    yield ("B", k * n + j)
                    yield ("C", i * n + j)
    else:
        for ii in range(0, n, T):
            for kk in range(0, n, T):
                for jj in range(0, n, T):
                    for i in range(ii, ii + T):
                        for k in range(kk, kk + T):
                            yield ("A", i * n + k)
                            for j in range(jj, jj + T):
                                yield ("B", k * n + j)
                                yield ("C", i * n + j)


def misses(order, n, line_words=8, n_lines=32, T=4):
    cache = OrderedDict()
    miss = 0
    for mat, idx in trace(order, n, T):
        key = (mat, idx // line_words)
        if key in cache:
            cache.move_to_end(key)
        else:
            miss += 1
            cache[key] = True
            if len(cache) > n_lines:
                cache.popitem(last=False)
    return miss


print(misses("ijk", 16), misses("ikj", 16))
