class Ctx:
    def __init__(self, b, t, dim):
        self.blockIdx, self.threadIdx, self.blockDim = b, t, dim


class Global:
    """Global bellek: her okumayı sayar."""
    def __init__(self, data):
        self.data, self.reads = list(data), 0

    def __getitem__(self, i):
        self.reads += 1
        return self.data[i]

    def __setitem__(self, i, v):
        self.data[i] = v


# ---- senin kodun ----

def launch(kernel, grid, block, *args):
    ...


def vec_add(ctx, x, y, out, n):
    ...


def matmul_naive(A, B, n):
    ...


def matmul_tiled(A, B, n, T):
    ...


n = 1000
x, y, out = Global(range(n)), Global(range(n)), Global([0] * n)
launch(vec_add, (n + 255) // 256, 256, x, y, out, n)
print(out.data[:5], x.reads)                     # [0, 2, 4, 6, 8] 1000
A, B = Global(range(16)), Global(range(16))
print(matmul_tiled(A, B, 4, 2)[:4], A.reads + B.reads)   # 64 okuma
