def q_values(P, V, s, nA, gamma):
    return [sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in P[s][a]) for a in range(nA)]


def value_iteration(P, nS, nA, gamma, tol=1e-10):
    V = [0.0] * nS
    while True:
        newV = [max(q_values(P, V, s, nA, gamma)) for s in range(nS)]
        delta = max(abs(a - b) for a, b in zip(newV, V))
        V = newV
        if delta < tol:
            break
    policy = []
    for s in range(nS):
        q = q_values(P, V, s, nA, gamma)
        policy.append(q.index(max(q)))
    return V, policy


def policy_evaluation(P, policy, nS, gamma, tol=1e-10):
    V = [0.0] * nS
    while True:
        newV = [sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in P[s][policy[s]]) for s in range(nS)]
        delta = max(abs(a - b) for a, b in zip(newV, V))
        V = newV
        if delta < tol:
            return V


P = {0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, False)]},
     1: {0: [(1.0, 1, 2.0, False)], 1: [(1.0, 0, 0.0, False)]}}
print(value_iteration(P, 2, 2, 0.5))     # ([3, 4], [1, 0])
