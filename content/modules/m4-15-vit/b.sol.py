def n_patches(H, W, P):
    return (H // P) * (W // P)


def vit_params(D, L, M, P, C, img, classes):
    N = n_patches(img, img, P)
    layer = 4 * (D * D + D) + D * M + M + M * D + D + 4 * D
    return P * P * C * D + D + D + (N + 1) * D + L * layer + 2 * D + D * classes + classes


def attn_flops(T, D):
    return 2 * T * T * D


def patchify(img, P):
    H, W = len(img), len(img[0])
    out = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            out.append([img[i + a][j + b] for a in range(P) for b in range(P)])
    return out
