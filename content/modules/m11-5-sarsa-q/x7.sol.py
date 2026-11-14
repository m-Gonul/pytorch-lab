import random


def q_target(states, actions, rewards, Q, pi, behavior, tau, n, kind="is", sigma=0.5, gamma=1.0):
    T = len(rewards)
    h = min(tau + n, T)
    value = lambda k: sum(p * q for p, q in zip(pi[states[k]], Q[states[k]]))
    selected = lambda k: Q[states[k]][actions[k]]
    ratio = lambda k: pi[states[k]][actions[k]] / behavior[states[k]][actions[k]]
    if kind in ("sarsa", "is", "expected"):
        G = 0.0 if h == T else (value(h) if kind == "expected" else selected(h))
        for k in range(h - 1, tau - 1, -1):
            G = rewards[k] + gamma * G
        weight = 1.0
        end = h - 1 if kind == "expected" else min(h, T - 1)
        if kind != "sarsa":
            for k in range(tau + 1, end + 1):
                weight *= ratio(k)
        return G, weight
    G = 0.0 if h == T else selected(h)
    for k in range(h - 1, tau - 1, -1):
        if k + 1 == T:
            G = rewards[k]
            continue
        j = k + 1
        if kind == "cv":
            coefficient = ratio(j)
        elif kind == "tree":
            coefficient = pi[states[j]][actions[j]]
        elif kind == "sigma":
            coefficient = sigma * ratio(j) + (1 - sigma) * pi[states[j]][actions[j]]
        else:
            raise ValueError("unknown return kind")
        G = rewards[k] + gamma * (value(j) + coefficient * (G - selected(j)))
    return G, 1.0


def learn_q_chain(episodes=10000, n=4, alpha=0.05, kind="sigma", sigma=0.5, seed=0):
    rng = random.Random(seed)
    Q = [[0.0, 0.0] for _ in range(5)]
    pi = [[0.75, 0.25] for _ in range(5)]
    behavior = [[0.5, 0.5] for _ in range(5)]
    for _ in range(episodes):
        states = list(range(5))
        actions = [int(rng.random() >= 0.5) for _ in range(4)]
        rewards = [0.0, 0.0, 0.0, 2.0 if actions[-1] == 0 else 0.0]
        for tau in range(4):
            G, weight = q_target(states, actions, rewards, Q, pi, behavior, tau, n, kind, sigma)
            s, a = states[tau], actions[tau]
            Q[s][a] += alpha * weight * (G - Q[s][a])
    return Q


if __name__ == "__main__":
    print(learn_q_chain())
