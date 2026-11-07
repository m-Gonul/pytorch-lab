import math


def arange(n):
    ...


def load(mem, offs, mask, other=0.0):
    ...


def store(mem, offs, vals, mask):
    ...


def launch(program, grid, *args):
    ...


def add_program(pid, x, y, out, n, BLOCK):
    ...


def softmax_program(pid, X, Y, n_cols, BLOCK):
    ...


n = 1000
x, y, out = list(range(n)), [1.0] * n, [None] * n
launch(add_program, (n + 255) // 256, x, y, out, n, 256)
print(out[:3], out[-1])
X = [-5.0, -6.0, -7.0, 1.0, 2.0, 3.0]; Y = [None] * 6
launch(softmax_program, 2, X, Y, 3, 4)
print(Y)
