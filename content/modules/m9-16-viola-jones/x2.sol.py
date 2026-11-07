from skimage.feature import haar_like_feature_coord


def count_type(W, H, nx, ny):
    return sum((W - w + 1) * (H - h + 1) for w in range(nx, W + 1, nx) for h in range(ny, H + 1, ny))


def total_features(W, H):
    return sum(count_type(W, H, nx, ny) for nx, ny in ((2, 1), (1, 2), (3, 1), (1, 3), (2, 2)))


print(total_features(24, 24))
