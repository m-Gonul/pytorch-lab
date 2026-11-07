import numpy as np


def fit_model(S_, A_, R_, S2_, nS, nA):
    ...


def plan(P, R, gamma, n_iter=500):
    ...


def q_learning_replay(S_, A_, R_, S2_, nS, nA, gamma, epochs=1, alpha=0.1):
    ...


def policy_value(P, R, pi, gamma):
    ...


rng = np.random.default_rng(0)
P = rng.random((2, 5, 5)); P /= P.sum(-1, keepdims=True); R = rng.random((5, 2))
S_ = rng.integers(0, 5, 100); A_ = rng.integers(0, 2, 100)
S2_ = np.array([rng.choice(5, p=P[a, s]) for s, a in zip(S_, A_)]); R_ = R[S_, A_]
Ph, Rh = fit_model(S_, A_, R_, S2_, 5, 2)
print(np.abs(Ph - P).max())
