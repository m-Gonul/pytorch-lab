def q_values(P, V, s, nA, gamma):
    ...


def value_iteration(P, nS, nA, gamma, tol=1e-10):
    ...


def policy_evaluation(P, policy, nS, gamma, tol=1e-10):
    ...


P = {0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, False)]},
     1: {0: [(1.0, 1, 2.0, False)], 1: [(1.0, 0, 0.0, False)]}}
print(value_iteration(P, 2, 2, 0.5))     # ([3, 4], [1, 0])
