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
    for b in range(grid):
        for t in range(block):
            kernel(Ctx(b, t, block), *args)


def vec_add(ctx, x, y, out, n):
    i = ctx.blockIdx * ctx.blockDim + ctx.threadIdx
    if i < n:
        out[i] = x[i] + y[i]


def matmul_naive(A, B, n):
    C = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i * n + k] * B[k * n + j]
            C[i * n + j] = s
    return C


def matmul_tiled(A, B, n, T):
    C = [0.0] * (n * n)
    nb = n // T
    for bi in range(nb):
        for bj in range(nb):
            acc = [[0.0] * T for _ in range(T)]
            for bk in range(nb):
                As = [[A[(bi * T + r) * n + bk * T + c] for c in range(T)] for r in range(T)]
                Bs = [[B[(bk * T + r) * n + bj * T + c] for c in range(T)] for r in range(T)]
                for r in range(T):
                    for c in range(T):
                        acc[r][c] += sum(As[r][k] * Bs[k][c] for k in range(T))
            for r in range(T):
                for c in range(T):
                    C[(bi * T + r) * n + bj * T + c] = acc[r][c]
    return C


n = 1000
x, y, out = Global(range(n)), Global(range(n)), Global([0] * n)
launch(vec_add, (n + 255) // 256, 256, x, y, out, n)
print(out.data[:5], x.reads)                     # [0, 2, 4, 6, 8] 1000
A, B = Global(range(16)), Global(range(16))
print(matmul_tiled(A, B, 4, 2)[:4], A.reads + B.reads)   # 64 okuma
