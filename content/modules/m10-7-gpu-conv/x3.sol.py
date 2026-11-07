def traffic_2d(H, W, K, T):
    tiles = (H // T) * (W // T)
    reads = tiles * (T + K - 1) ** 2
    return reads, H * W, H * W * K * K


def traffic_separable(H, W, K, T):
    tiles = (H // T) * (W // T)
    r1 = tiles * T * (T + K - 1)
    w1 = H * W * 4
    r2 = tiles * (T + K - 1) * T * 4
    return r1 + r2, w1 + H * W, 2 * H * W * K


print(traffic_2d(1024, 1024, 21, 16), traffic_separable(1024, 1024, 21, 16))
