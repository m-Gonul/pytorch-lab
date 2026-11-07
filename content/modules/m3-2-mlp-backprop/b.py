def linear_forward(x, W, b):
    ...


def linear_backward(dz, x, W):
    ...


def relu_forward(z):
    ...


def relu_backward(dh, z):
    ...


def mlp_loss(x, y, W1, b1, w2, b2):
    ...


L, g = mlp_loss([1, 2], 1, [[1, 2], [-1, 1]], [0, 0], [1, -2], 0)
print(L, g)    # 2.0, W1: [[2, 4], [-4, -8]], x: [6, 0]
