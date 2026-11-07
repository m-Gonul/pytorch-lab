def contiguous_strides(shape):
    st, acc = [], 1
    for n in reversed(shape):
        st.append(acc)
        acc *= n
    return tuple(reversed(st))


def get(data, strides, offset, index):
    addr = offset
    for i, s in zip(index, strides):
        addr += i * s
    return data[addr]


def transpose(shape, strides, i, j):
    sh, st = list(shape), list(strides)
    sh[i], sh[j] = sh[j], sh[i]
    st[i], st[j] = st[j], st[i]
    return tuple(sh), tuple(st)


def materialize(data, shape, strides, offset=0):
    if len(shape) == 0:
        return data[offset]
    return [materialize(data, shape[1:], strides[1:], offset + k * strides[0]) for k in range(shape[0])]


def windows(n, w):
    """Uzunluk n dizide w'lik kayan pencere view'u → (shape, strides)."""
    return (n - w + 1, w), (1, 1)


data = [0, 1, 2, 3, 4, 5]                      # A = [[0,1,2],[3,4,5]]
st = contiguous_strides((2, 3))
print(st, get(data, st, 0, (1, 2)))             # (3, 1) 5
sh, st2 = transpose((2, 3), st, 0, 1)
print(materialize(data, sh, st2))               # [[0,3],[1,4],[2,5]]
print(materialize(data, *windows(6, 3)))        # [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
