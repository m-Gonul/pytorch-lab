import random
import torch


def differential_torch_step(Q, rho, state, action, reward, next_state, next_action,
                            alpha=0.01, beta=0.01):
    prediction = Q[state, action]
    target = reward - rho + Q[next_state, next_action].detach()
    delta = float((target - prediction).detach())
    loss = 0.5 * (target - prediction).square()
    gradient, = torch.autograd.grad(loss, Q)
    updated = (Q - alpha * gradient).detach().requires_grad_()
    return updated, rho + beta * delta


def torch_sarsa(steps=5000, alpha=0.01, beta=0.01, epsilon=0.1, seed=0):
    rng = random.Random(seed)
    Q = torch.zeros(44, 2, dtype=torch.float64, requires_grad=True)
    visits = [0] * 44

    def choose(free, priority):
        if free == 0:
            return 0
        if rng.random() < epsilon:
            return rng.randrange(2)
        row = Q[free * 4 + priority].detach().tolist()
        return rng.choice([a for a in range(2) if row[a] == max(row)])

    free, priority, rho, total = 10, rng.randrange(4), 0.0, 0.0
    action = choose(free, priority)
    for _ in range(steps):
        accepted = int(action == 1 and free > 0)
        reward = float(accepted * (1 << priority))
        after = free - accepted
        nf = after + sum(rng.random() < .06 for _ in range(10-after))
        np_ = rng.randrange(4)
        na = choose(nf, np_)
        state = 4 * free + priority
        Q, rho = differential_torch_step(Q, rho, state, action, reward,
                                         4 * nf + np_, na, alpha, beta)
        visits[state] += 1
        total += reward
        free, priority, action = nf, np_, na
    return dict(Q=Q.detach(), rho=rho, reward_mean=total/steps, visits=visits)


print(torch_sarsa(1000)["reward_mean"])
