def semi_gradient_n(episodes, alpha, n, k=20, gamma=1.0):
    if n <= 0:
        raise ValueError("n pozitif olmalı")
    w = [0.0] * k
    group = lambda s: (s - 1) * k // 1000
    for ep in episodes:
        T = len(ep)
        for t, (state, _, _) in enumerate(ep):
            stop = min(T, t + n)
            G = sum(gamma ** (j - t) * ep[j][1] for j in range(t, stop))
            if stop < T:
                G += gamma ** n * w[group(ep[stop][0])]
            i = group(state)
            w[i] += alpha * (G - w[i])
    return w
