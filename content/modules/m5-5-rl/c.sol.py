import numpy as np


def bellman_q(V, P, R, gamma):
    return R + gamma * np.einsum("sat,t->sa", P, V)


def policy_eval_exact(P, R, pi, gamma):
    Pp = np.einsum("sa,sat->st", pi, P)
    Rp = (pi * R).sum(1)
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * Pp, Rp)


def policy_iteration(P, R, gamma):
    S, A = R.shape
    pol = np.zeros(S, int)
    while True:
        V = policy_eval_exact(P, R, np.eye(A)[pol], gamma)
        new = bellman_q(V, P, R, gamma).argmax(1)
        if np.array_equal(new, pol):
            return V, pol
        pol = new


P = np.zeros((2, 2, 2)); P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 1] = P[1, 1, 0] = 1
R = np.array([[0.0, 1.0], [2.0, 0.0]])
print(policy_iteration(P, R, 0.5))   # V = [3, 4], politika [1, 0]
