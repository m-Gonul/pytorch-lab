import numpy as np
import torch
from scipy.stats import poisson


def location(lam_req, lam_ret, N=20):
    T = np.zeros((N + 1, N + 1))
    r = np.zeros(N + 1)
    pr = poisson.pmf(np.arange(N + 1), lam_ret)
    pr[-1] = poisson.sf(N - 1, lam_ret)
    for n in range(N + 1):
        pq = poisson.pmf(np.arange(n + 1), lam_req)
        pq[-1] = poisson.sf(n - 1, lam_req)
        r[n] = 10 * (pq * np.arange(n + 1)).sum()
        for k, p in enumerate(pq):
            np.add.at(T[n], np.minimum(n - k + np.arange(N + 1), N), p * pr)
    return T, r


def jack_pi(gamma=0.9):
    N = 20
    T1, r1 = (torch.tensor(x) for x in location(3, 3))
    T2, r2 = (torch.tensor(x) for x in location(4, 2))
    n1, n2 = torch.meshgrid(torch.arange(N + 1), torch.arange(N + 1), indexing="ij")
    acts = torch.arange(-5, 6)

    def post(a):
        return (a <= n1) & (-a <= n2), (n1 - a).clamp(max=N), (n2 + a).clamp(max=N)

    pi = torch.zeros(N + 1, N + 1, dtype=torch.long)
    it = 0
    while True:
        ok, m1, m2 = post(pi)
        Rp = r1[m1] + r2[m2] - 2 * pi.abs()
        Pp = (T1[m1][:, :, :, None] * T2[m2][:, :, None, :]).reshape(441, 441)
        V = torch.linalg.solve(torch.eye(441, dtype=torch.float64) - gamma * Pp, Rp.reshape(-1)).reshape(21, 21)
        it += 1
        Q = torch.full((11, 21, 21), -float("inf"), dtype=torch.float64)
        for i, a in enumerate(acts.tolist()):
            ok, m1, m2 = post(torch.full_like(n1, a))
            q = r1[m1] + r2[m2] - 2 * abs(a) + gamma * torch.einsum("xyi,ij,xyj->xy", T1[m1], V, T2[m2])
            Q[i] = torch.where(ok, q, Q[i])
        new = acts[Q.argmax(0)]
        if torch.equal(new, pi):
            return V, pi, it
        pi = new


V, pi, it = jack_pi()
print(it, pi[20, 0], round(float(V[0, 0]), 1))
