def grid_step(s, a):
    ...


def evaluate(pi, gamma=1.0, theta=1e-10):
    ...


def improve(V, gamma=1.0):
    ...


def policy_iteration(gamma=1.0):
    ...


V, sweeps = evaluate([[0.25] * 4 for _ in range(16)])
print([round(v) for v in V])
