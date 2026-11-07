import random
import math


def episodes19(k, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        s = 10
        st, rw = [s], []
        while s not in (0, 20):
            s += 1 if rng.random() < 0.5 else -1
            st.append(s); rw.append(1.0 if s == 20 else (-1.0 if s == 0 else 0.0))
        out.append((st, rw))
    return out


def nstep_td(episodes, n, alpha, N=19, gamma=1.0):
    V = [0.0] * (N + 2)
    for st, rw in episodes:
        T = len(rw)
        for tau in range(T):
            G = sum(gamma ** (i - tau) * rw[i] for i in range(tau, min(tau + n, T)))
            if tau + n < T:
                G += gamma ** n * V[st[tau + n]]
            V[st[tau]] += alpha * (G - V[st[tau]])
    true = [i / 10 - 1 for i in range(1, 20)]
    return math.sqrt(sum((V[i] - true[i - 1]) ** 2 for i in range(1, 20)) / 19)


print(nstep_td(episodes19(10), 4, 0.4))
