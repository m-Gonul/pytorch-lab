import torch


def random_sparse_mdp(S, A, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, S, (A, S, k), generator=g)
    prob = torch.rand(A, S, k, generator=g, dtype=torch.float64)
    prob /= prob.sum(-1, keepdim=True)
    R = torch.randn(S, A, generator=g, dtype=torch.float64)
    return idx, prob, R


def eval_direct(Pi_dense, r, gamma):
    S = r.shape[0]
    return torch.linalg.solve(torch.eye(S, dtype=r.dtype) - gamma * Pi_dense, r)


def eval_iterative(idx, prob, r, gamma, tol=1e-10):
    V = torch.zeros_like(r)
    n = 0
    while True:
        Vn = r + gamma * (prob * V[idx]).sum(-1)
        n += 1
        if (Vn - V).abs().max() <= tol:
            return Vn, n
        V = Vn


def mpi_sparse(idx, prob, R, gamma, k, eps=1e-8):
    A, S, _ = idx.shape
    ar = torch.arange(S)
    V = torch.zeros(S, dtype=R.dtype)
    it = 0
    while True:
        Q = R + gamma * (prob * V[idx]).sum(-1).T
        pi, Vn = Q.argmax(1), Q.max(1).values
        it += 1
        if (Vn - V).abs().max() <= eps:
            return Vn, pi, it
        V = Vn
        ip, pp, rp = idx[pi, ar], prob[pi, ar], R[ar, pi]
        for _ in range(k - 1):
            V = rp + gamma * (pp * V[ip]).sum(-1)


idx, prob, R = random_sparse_mdp(100, 3, 5)
print(idx.shape, prob.shape, R.shape)
