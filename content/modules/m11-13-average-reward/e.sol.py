import numpy as np
import torch


def batched_access(n_envs=64, steps=5000, alpha=0.01, beta=0.01, epsilon=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = torch.zeros(n_envs, 44, 2, dtype=torch.float64)
    rho, total = torch.zeros(n_envs, dtype=torch.float64), torch.zeros(n_envs, dtype=torch.float64)
    visits = torch.zeros(n_envs, 44, dtype=torch.int64)
    batch = torch.arange(n_envs)
    free, priority = np.full(n_envs, 10), rng.integers(0, 4, n_envs)

    def choose(f, j):
        row = Q[batch, torch.from_numpy(4*f+j)].numpy()
        greedy = (row[:, 1] > row[:, 0]).astype(np.int64)
        ties = row[:, 0] == row[:, 1]
        random_action = rng.integers(0, 2, n_envs)
        action = np.where(ties | (rng.random(n_envs) < epsilon), random_action, greedy)
        return np.where(f == 0, 0, action)

    action = choose(free, priority)
    server = np.arange(10)[None, :]
    for _ in range(steps):
        accepted = action * (free > 0)
        reward = torch.from_numpy((accepted * (1 << priority)).astype(float))
        after = free - accepted
        busy = 10 - after
        released = ((rng.random((n_envs, 10)) < .06) & (server < busy[:, None])).sum(axis=1)
        nf, np_ = after + released, rng.integers(0, 4, n_envs)
        na = choose(nf, np_)
        state, next_state = torch.from_numpy(4*free+priority), torch.from_numpy(4*nf+np_)
        a, next_a = torch.from_numpy(action), torch.from_numpy(na)
        delta = reward - rho + Q[batch, next_state, next_a] - Q[batch, state, a]
        rho += beta * delta
        Q[batch, state, a] += alpha * delta
        visits[batch, state] += 1
        total += reward
        free, priority, action = nf, np_, na
    return dict(Q=Q, rho=rho, reward_mean=total/steps, visits=visits)


print(batched_access(16, 1000)["reward_mean"].mean().item())
