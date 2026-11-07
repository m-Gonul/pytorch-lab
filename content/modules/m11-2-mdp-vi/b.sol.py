def backup(P, R, V, gamma, s):
    return [R[s][a] + gamma * sum(p * v for p, v in zip(P[a][s], V)) for a in range(len(P))]


def finite_vi(P, R, VT, h, gamma=1.0):
    Vs = [None] * (h + 1)
    pis = [None] * h
    Vs[h] = list(VT)
    for t in range(h - 1, -1, -1):
        V, pi = [], []
        for s in range(len(VT)):
            q = backup(P, R, Vs[t + 1], gamma, s)
            a = q.index(max(q))
            V.append(q[a]); pi.append(a)
        Vs[t], pis[t] = V, pi
    return Vs, pis


def retail_mdp():
    S = 4
    P = [[[0.0] * S for _ in range(S)] for _ in range(2)]
    for a, sale in enumerate((0.3, 0.8)):
        P[a][0][1] = 1.0
        for k in range(1, S):
            P[a][k][k - 1] = sale
            P[a][k][k] = 1 - sale
    cost = [-5, -1, -1, -2]
    R = [[c, c] for c in cost]
    return P, R, [0, -4, -8, -12]


P, R, VT = retail_mdp()
Vs, pis = finite_vi(P, R, VT, 5)
print(Vs[4], pis[3])
