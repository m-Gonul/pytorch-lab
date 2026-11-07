def backup(P, R, V, gamma, s):
    ...


def finite_vi(P, R, VT, h, gamma=1.0):
    ...


def retail_mdp():
    ...


P, R, VT = retail_mdp()
Vs, pis = finite_vi(P, R, VT, 5)
print(Vs[4], pis[3])
