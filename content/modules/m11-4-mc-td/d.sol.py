import torch


def _v(V, s, terminal):
    return 0.0 if terminal else V[s]


def lambda_returns(states, rewards, V, gamma, lam):
    T = len(rewards)
    G = torch.zeros(T, dtype=V.dtype)
    nxt = 0.0
    for t in range(T - 1, -1, -1):
        vn = _v(V, states[t + 1], t + 1 == T)
        nxt = rewards[t] + gamma * ((1 - lam) * vn + lam * nxt)
        G[t] = nxt
    return G


def offline_forward(states, rewards, V, alpha, gamma, lam):
    G = lambda_returns(states, rewards, V, gamma, lam)
    dV = torch.zeros_like(V)
    for t in range(len(rewards)):
        dV[states[t]] += alpha * (G[t] - V[states[t]])
    return V + dV


def offline_traces(states, rewards, V, alpha, gamma, lam):
    e = torch.zeros_like(V)
    dV = torch.zeros_like(V)
    T = len(rewards)
    for t in range(T):
        delta = rewards[t] + gamma * _v(V, states[t + 1], t + 1 == T) - V[states[t]]
        e = gamma * lam * e
        e[states[t]] += 1
        dV += alpha * delta * e
    return V + dV


def online_traces(states, rewards, V, alpha, gamma, lam):
    V = V.clone()
    e = torch.zeros_like(V)
    T = len(rewards)
    for t in range(T):
        delta = rewards[t] + gamma * _v(V, states[t + 1], t + 1 == T) - V[states[t]]
        e = gamma * lam * e
        e[states[t]] += 1
        V = V + alpha * delta * e
    return V


V = torch.zeros(7, dtype=torch.float64)
print(offline_traces([3, 4, 5, 6], [0.0, 0.0, 1.0], V, 0.1, 1.0, 0.8))
