def semi_gradient(w, alpha, gamma, n):
    out = []
    for _ in range(n):
        w = w + alpha * (gamma * 2 * w - w) * 1
        out.append(w)
    return out


def residual_gradient(w, alpha, gamma, n):
    out = []
    for _ in range(n):
        w = w - alpha * (2 * gamma - 1) ** 2 * w
        out.append(w)
    return out


def on_policy(w, alpha, gamma, n):
    out = []
    for _ in range(n):
        w = w + alpha * (gamma * 2 * w - w) * 1
        w = w + alpha * (0 - 2 * w) * 2
        out.append(w)
    return out


print(semi_gradient(1.0, 0.1, 0.9, 10)[-1])
