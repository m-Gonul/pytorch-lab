import math


def mse(x, y):
    return sum((a - b) ** 2 for a, b in zip(x, y)) / len(x)


def psnr(x, y, R=255):
    m = mse(x, y)
    return float("inf") if m == 0 else 10 * math.log10(R * R / m)


def _stats(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((a - mx) ** 2 for a in x) / (n - 1)
    vy = sum((b - my) ** 2 for b in y) / (n - 1)
    cxy = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n - 1)
    return mx, my, vx, vy, cxy


def ssim_global(x, y, L=255, K1=0.01, K2=0.03):
    mx, my, vx, vy, cxy = _stats(x, y)
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    return (2 * mx * my + C1) * (2 * cxy + C2) / ((mx * mx + my * my + C1) * (vx + vy + C2))


def ssim_components(x, y, L=255, K1=0.01, K2=0.03):
    mx, my, vx, vy, cxy = _stats(x, y)
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    C3 = C2 / 2
    sx, sy = math.sqrt(vx), math.sqrt(vy)
    return ((2 * mx * my + C1) / (mx * mx + my * my + C1), (2 * sx * sy + C2) / (vx + vy + C2), (cxy + C3) / (sx * sy + C3))


print(ssim_global([1, 2, 3, 4], [2, 3, 4, 5]))
