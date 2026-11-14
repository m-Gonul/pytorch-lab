import random
import math


def episodes19(k, seed=0, N=19, left=-1.0):
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        s = (N + 1) // 2
        st, rw = [s], []
        while s not in (0, N + 1):
            s += 1 if rng.random() < 0.5 else -1
            st.append(s); rw.append(1.0 if s == N + 1 else (left if s == 0 else 0.0))
        out.append((st, rw))
    return out


def walk_values(N=19, left=-1.0, gamma=1.0):
    # Solve the discounted tridiagonal Bellman equations, terminal V=0.
    diagonal = [1.0] * N
    rhs = [0.0] * N
    rhs[0] += left / 2
    rhs[-1] += 0.5
    lower = upper = -gamma / 2
    for i in range(1, N):
        factor = lower / diagonal[i - 1]
        diagonal[i] -= factor * upper
        rhs[i] -= factor * rhs[i - 1]
    true = [0.0] * N
    true[-1] = rhs[-1] / diagonal[-1]
    for i in range(N - 2, -1, -1):
        true[i] = (rhs[i] - upper * true[i + 1]) / diagonal[i]
    return true


def nstep_td(episodes, n, alpha, N=19, gamma=1.0, left=-1.0, initial=0.0, history=False):
    V = [0.0] + [initial] * N + [0.0]
    true = walk_values(N, left, gamma)
    errors = []
    for st, rw in episodes:
        T = len(rw)
        for tau in range(T):
            G = sum(gamma ** (i - tau) * rw[i] for i in range(tau, min(tau + n, T)))
            if tau + n < T:
                G += gamma ** n * V[st[tau + n]]
            V[st[tau]] += alpha * (G - V[st[tau]])
        errors.append(math.sqrt(sum((V[i] - true[i - 1]) ** 2 for i in range(1, N + 1)) / N))
    return errors if history else errors[-1]


print(nstep_td(episodes19(10), 4, 0.4))
