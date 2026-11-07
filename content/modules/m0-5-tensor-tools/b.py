def contiguous_strides(shape):
    ...


def get(data, strides, offset, index):
    ...


def transpose(shape, strides, i, j):
    ...


def materialize(data, shape, strides, offset=0):
    ...


def windows(n, w):
    """Uzunluk n dizide w'lik kayan pencere view'u → (shape, strides)."""
    ...


data = [0, 1, 2, 3, 4, 5]                      # A = [[0,1,2],[3,4,5]]
st = contiguous_strides((2, 3))
print(st, get(data, st, 0, (1, 2)))             # (3, 1) 5
sh, st2 = transpose((2, 3), st, 0, 1)
print(materialize(data, sh, st2))               # [[0,3],[1,4],[2,5]]
print(materialize(data, *windows(6, 3)))        # [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
