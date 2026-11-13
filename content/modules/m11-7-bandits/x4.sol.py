import numpy as np


def bandit_sweep(n_runs, n_steps, configs, seed=0, drift=0.01, noise=1, means=None):
    if n_runs < 1 or n_steps < 2 or not configs:
        raise ValueError("Pozitif tekrar, en az iki adım ve ayar listesi gerekiyor.")
    rng = np.random.default_rng(seed)
    kinds = np.array([c[0] for c in configs])
    if not set(kinds) <= {"epsilon", "constant", "ucb", "optimistic", "gradient"}:
        raise ValueError("Bilinmeyen yöntem.")
    parameters = np.array([c[1] for c in configs], dtype=float)
    initial = np.zeros(10) if means is None else np.asarray(means, dtype=float)
    runs, count, arms = n_runs, len(configs), len(initial)
    truth = np.broadcast_to(initial, (runs, arms)).copy()
    Q = np.zeros((runs, count, arms))
    Q[:, kinds == "optimistic", :] = parameters[kinds == "optimistic"][None, :, None]
    H, N = np.zeros_like(Q), np.zeros_like(Q, dtype=int)
    baseline = np.zeros((runs, count))
    total = np.zeros(count)
    rr, cc = np.arange(runs)[:, None], np.arange(count)[None, :]
    eps = np.where(np.isin(kinds, ["epsilon", "constant"]), parameters, 0)
    gradient = kinds == "gradient"
    ucb = kinds == "ucb"
    constant = np.isin(kinds, ["constant", "optimistic"])
    tail = n_steps // 2
    for t in range(1, n_steps + 1):
        scores = Q.copy()
        if ucb.any():
            n = N[:, ucb, :]
            scores[:, ucb, :] += parameters[ucb][None, :, None] * np.sqrt(np.log(t) / np.maximum(n, 1))
            scores[:, ucb, :] = np.where(n == 0, np.inf, scores[:, ucb, :])
        ties = scores == scores.max(axis=2, keepdims=True)
        actions = (ties * rng.random(Q.shape)).argmax(axis=2)
        actions = np.where(rng.random((runs, count)) < eps, rng.integers(arms, size=(runs, count)), actions)
        probs = np.exp(H - H.max(axis=2, keepdims=True))
        probs /= probs.sum(axis=2, keepdims=True)
        if gradient.any():
            draws = rng.random((runs, int(gradient.sum()), 1))
            actions[:, gradient] = (probs[:, gradient, :].cumsum(axis=2) < draws).sum(axis=2).clip(max=arms-1)
        reward = truth[rr, actions] + noise * rng.normal(size=(runs, count))
        if t > tail:
            total += reward.mean(axis=0) / (n_steps-tail)
        N[rr, cc, actions] += 1
        old = Q[rr, cc, actions]
        rate = np.where(constant, 0.1, 1 / N[rr, cc, actions])
        Q[rr, cc, actions] += rate * (reward-old)
        baseline += (reward-baseline)/t
        onehot = np.eye(arms)[actions]
        H[:, gradient, :] += parameters[gradient][None, :, None] * (reward-baseline)[:, gradient, None] * (onehot-probs)[:, gradient, :]
        truth += drift * rng.normal(size=truth.shape)
    return total
