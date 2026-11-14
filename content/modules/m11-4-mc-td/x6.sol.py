import random


def compare_chain(runs=1000, episodes=100, n=4, alpha=0.1, p=0.9, seed=0):
    rng = random.Random(seed)
    mse = [[0.0] * episodes for _ in range(2)]
    for _ in range(runs):
        tables = [[0.0] * 4 for _ in range(2)]
        for episode in range(episodes):
            ratios = [(p if rng.random() < 0.5 else 1 - p) / 0.5 for _ in range(4)]
            for method, V in enumerate(tables):
                for tau in range(4):
                    h = min(tau + n, 4)
                    if method == 0:
                        G = 1.0 if h == 4 else V[h]
                        weight = 1.0
                        for k in range(tau, h):
                            weight *= ratios[k]
                        V[tau] += alpha * weight * (G - V[tau])
                    else:
                        G = 0.0 if h == 4 else V[h]
                        for k in range(h - 1, tau - 1, -1):
                            G = ratios[k] * ((1.0 if k == 3 else 0.0) + G) + (1 - ratios[k]) * V[k]
                        V[tau] += alpha * (G - V[tau])
                mse[method][episode] += sum((value - 1) ** 2 for value in V) / 4 / runs
    return {"ordinary": mse[0], "cv": mse[1]}


if __name__ == "__main__":
    out = compare_chain()
    print({k: v[-1] for k, v in out.items()})
