def inverted_residual_cost(c, H, t, k=3):
    e = t * c
    p = c * e + k * k * e + e * c
    return p, H * H * p


def bottleneck_cost(c, H, r, k=3):
    m = c // r
    p = c * m + k * k * m * m + m * c
    return p, H * H * p


print(inverted_residual_cost(32, 28, 6))
