import random


def gain_step(rho, delta, n, beta, trace, debias=True):
    if n < 1 or not 0 < beta <= 1:
        raise ValueError("n pozitif, beta (0,1] içinde olmalı")
    if debias:
        trace += beta * (1 - trace)
        rate = beta / (trace * n)
    else:
        rate = beta
    return rho + rate * delta, trace


def differential_n_sarsa(steps=5000, n=2, alpha=.01, beta=.01, epsilon=.1,
                         seed=0, debias=True, initial_gain=0.0):
    if n < 1:
        raise ValueError("n pozitif olmalı")
    rng = random.Random(seed)
    Q = [[[0., 0.] for _ in range(4)] for _ in range(11)]
    states, actions, rewards = [None]*(n+1), [0]*(n+1), [0.]*(n+1)
    visits = [[0]*4 for _ in range(11)]

    def choose(f, j):
        if f == 0:
            return 0
        if rng.random() < epsilon:
            return rng.randrange(2)
        m = max(Q[f][j])
        return rng.choice([a for a in range(2) if Q[f][j][a] == m])

    states[0] = (10, rng.randrange(4))
    actions[0] = choose(*states[0])
    rho, trace, total, updates = float(initial_gain), 0., 0., 0
    for t in range(steps):
        free, priority = states[t % (n+1)]
        action = actions[t % (n+1)]
        accepted = int(action == 1 and free > 0)
        R = float(accepted * (1 << priority))
        after = free - accepted
        nf = after + sum(rng.random() < .06 for _ in range(10-after))
        np_ = rng.randrange(4)
        states[(t+1) % (n+1)] = (nf, np_)
        actions[(t+1) % (n+1)] = choose(nf, np_)
        rewards[(t+1) % (n+1)] = R
        total += R
        visits[free][priority] += 1
        tau = t-n+1
        if tau >= 0:
            f, j = states[tau % (n+1)]
            a = actions[tau % (n+1)]
            ff, jj = states[(tau+n) % (n+1)]
            aa = actions[(tau+n) % (n+1)]
            delta = sum(rewards[i % (n+1)] for i in range(tau+1, tau+n+1)) - n*rho
            delta += Q[ff][jj][aa] - Q[f][j][a]
            rho, trace = gain_step(rho, delta, n, beta, trace, debias)
            Q[f][j][a] += alpha * delta
            updates += 1
    return dict(Q=Q, rho=rho, trace=trace, reward_mean=total/steps,
                visits=visits, updates=updates)


print(differential_n_sarsa(2000)["rho"])
