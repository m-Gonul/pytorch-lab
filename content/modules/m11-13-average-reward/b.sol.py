import random


def access_step(free, priority, action, rng, p=0.06, release_new=True):
    accepted = int(bool(action) and free > 0)
    reward = accepted * (1 << priority)
    after = free - accepted
    busy = 10 - (after if release_new else free)
    next_free = after + sum(rng.random() < p for _ in range(busy))
    return next_free, rng.randrange(4), float(reward)


def differential_sarsa(steps=20000, alpha=0.01, beta=0.01, epsilon=0.1,
                       seed=0, p=0.06, release_new=True):
    rng = random.Random(seed)
    Q = [[[0.0, 0.0] for _ in range(4)] for _ in range(11)]
    visits = [[0] * 4 for _ in range(11)]

    def choose(f, j):
        if f == 0:
            return 0
        if rng.random() < epsilon:
            return rng.randrange(2)
        row = Q[f][j]
        best = max(row)
        return rng.choice([a for a in range(2) if row[a] == best])

    free, priority, rho, total = 10, rng.randrange(4), 0.0, 0.0
    action = choose(free, priority)
    for _ in range(steps):
        nf, np_, reward = access_step(free, priority, action, rng, p, release_new)
        na = choose(nf, np_)
        target = reward - rho + Q[nf][np_][na]
        delta = target - Q[free][priority][action]
        rho += beta * delta
        Q[free][priority][action] += alpha * delta
        visits[free][priority] += 1
        total += reward
        free, priority, action = nf, np_, na
    return dict(Q=Q, rho=rho, reward_mean=total / steps, visits=visits)


print(differential_sarsa(5000)["reward_mean"])
