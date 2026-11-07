def linear_forward(x, W, b):
    return [sum(w * xi for w, xi in zip(row, x)) + bi for row, bi in zip(W, b)]


def linear_backward(dz, x, W):
    dx = [sum(W[i][j] * dz[i] for i in range(len(W))) for j in range(len(x))]
    dW = [[g * xi for xi in x] for g in dz]
    return dx, dW, list(dz)


def relu_forward(z):
    return [v if v > 0 else 0.0 for v in z]


def relu_backward(dh, z):
    return [g if v > 0 else 0.0 for g, v in zip(dh, z)]


def mlp_loss(x, y, W1, b1, w2, b2):
    z = linear_forward(x, W1, b1)
    h = relu_forward(z)
    yh = sum(a * b for a, b in zip(w2, h)) + b2
    L = 0.5 * (yh - y) ** 2
    dy = yh - y
    dw2 = [dy * v for v in h]
    dh = [dy * w for w in w2]
    dz = relu_backward(dh, z)
    dx, dW1, db1 = linear_backward(dz, x, W1)
    return L, {"W1": dW1, "b1": db1, "w2": dw2, "b2": dy, "x": dx}


L, g = mlp_loss([1, 2], 1, [[1, 2], [-1, 1]], [0, 0], [1, -2], 0)
print(L, g)    # 2.0, W1: [[2, 4], [-4, -8]], x: [6, 0]
