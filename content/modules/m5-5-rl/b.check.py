import numpy as np
from labcheck import close, no_imports, need, ensure


def frozen_lake():
    grid = ["SFFF", "FHFH", "FFFH", "HFFG"]
    nS, nA = 16, 4
    moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    P = {s: {a: [] for a in range(nA)} for s in range(nS)}
    for s in range(nS):
        r, c = divmod(s, 4)
        for a in range(nA):
            if grid[r][c] in "HG":
                P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for b in ((a - 1) % 4, a, (a + 1) % 4):
                dr, dc = moves[b]
                nr, nc = min(max(r + dr, 0), 3), min(max(c + dc, 0), 3)
                s2 = nr * 4 + nc
                P[s][a].append((1 / 3, s2, float(grid[nr][nc] == "G"), grid[nr][nc] in "HG"))
    return P


def to_mats(P, nS, nA):
    T = np.zeros((nS, nA, nS)); R = np.zeros((nS, nA)); D = np.zeros((nS, nA, nS))
    for s in range(nS):
        for a in range(nA):
            for p, s2, r, d in P[s][a]:
                R[s, a] += p * r
                if not d:
                    T[s, a, s2] += p
    return T, R


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    P2 = {0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, False)]}, 1: {0: [(1.0, 1, 2.0, False)], 1: [(1.0, 0, 0.0, False)]}}

    def tiny():
        V, pi = need(ns, "value_iteration")(P2, 2, 2, 0.5)
        close(V, [3, 4], atol=1e-8); ensure(list(pi) == [1, 0], f"politika {pi}")
    R_.test("iki durumlu MDP (elle örnek)", tiny)
    P = frozen_lake()
    T, R = to_mats(P, 16, 4)

    def fl():
        V, pi = need(ns, "value_iteration")(P, 16, 4, 0.99)
        Vr = np.zeros(16)
        for _ in range(5000):
            Vr = (R + 0.99 * T @ Vr).max(1)
        close(V, Vr, atol=1e-6)
        Pp = T[np.arange(16), pi]; Rp = R[np.arange(16), pi]
        Vpi = np.linalg.solve(np.eye(16) - 0.99 * Pp, Rp)
        close(Vpi, Vr, atol=1e-6, what="açgözlü politikanın değeri = V*")
        return f"V*(başlangıç) = {V[0]:.4f}: hedefe varma olasılığı (γ = 0.99 ile indirimli)"
    R_.test("kaygan buzlu göl: V* ve optimal politika", fl)

    def pe():
        pol = [2] * 16
        V = need(ns, "policy_evaluation")(P, pol, 16, 0.9)
        Pp = T[np.arange(16), pol]; Rp = R[np.arange(16), pol]
        close(V, np.linalg.solve(np.eye(16) - 0.9 * Pp, Rp), atol=1e-6)
    R_.test("policy_evaluation (hep sağ)", pe)

    def qv():
        V = [0.5] * 16
        close(need(ns, "q_values")(P, V, 14, 4, 0.9), R[14] + 0.9 * T[14] @ np.array(V))
    R_.test("q_values (hedefe komşu durum)", qv)
