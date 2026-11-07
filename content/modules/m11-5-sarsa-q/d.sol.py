import torch


def q_value_iteration(P, R, gamma, tol=1e-10):
    Q = torch.zeros_like(R)
    while True:
        Qn = R + gamma * torch.einsum("ast,t->sa", P, Q.max(1).values)
        if (Qn - Q).abs().max() < tol:
            return Qn
        Q = Qn


def sync_q_learning(P, R, gamma, n_iter, rate="rescaled", seed=0):
    g = torch.Generator().manual_seed(seed)
    A, S, _ = P.shape
    probs = P.permute(1, 0, 2).reshape(S * A, S)
    Q = torch.zeros_like(R)
    for n in range(1, n_iter + 1):
        s2 = torch.multinomial(probs, 1, generator=g).view(S, A)
        alpha = 1.0 / n if rate == "1/n" else 1.0 / (1 + (1 - gamma) * n)
        Q = Q + alpha * (R + gamma * Q.max(1).values[s2] - Q)
    return Q


def greedy(Q):
    return Q.argmax(1)


g = torch.Generator().manual_seed(0)
P = torch.rand(3, 8, 8, generator=g, dtype=torch.float64); P /= P.sum(-1, keepdim=True)
R = torch.rand(8, 3, generator=g, dtype=torch.float64)
print((sync_q_learning(P, R, 0.9, 1000) - q_value_iteration(P, R, 0.9)).abs().max())
