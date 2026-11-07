def flops_factor(alpha, beta, gamma, phi):
    return (alpha * beta ** 2 * gamma ** 2) ** phi


def grid_candidates(step=0.05):
    vals = [round(1 + i * step, 4) for i in range(int(1 / step) + 1)]
    return [(a, b, g) for a in vals for b in vals for g in vals if abs(a * b * b * g * g - 2) <= 0.1]


def scale_model(base, alpha, beta, gamma, phi):
    d, w, r = base
    return round(d * alpha ** phi), round(w * beta ** phi), round(r * gamma ** phi)


print(flops_factor(1.2, 1.1, 1.15, 1))
