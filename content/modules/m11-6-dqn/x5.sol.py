def _linear_solve(A, b):
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for j in range(n):
        k = max(range(j, n), key=lambda i: abs(M[i][j]))
        if abs(M[k][j]) < 1e-12:
            raise ValueError("tekil sistem")
        M[k], M[j] = M[j], M[k]
        p = M[j][j]
        M[j] = [v / p for v in M[j]]
        for i in range(n):
            if i != j:
                c = M[i][j]
                M[i] = [a - c * v for a, v in zip(M[i], M[j])]
    return [row[-1] for row in M]


def batch_lstd(transitions, gamma, epsilon):
    if epsilon <= 0:
        raise ValueError("epsilon pozitif olmalı")
    if not transitions:
        return []
    d = len(transitions[0][0])
    A = [[epsilon * (i == j) for j in range(d)] for i in range(d)]
    b = [0.0] * d
    for x, reward, next_x in transitions:
        y = [x[j] - gamma * next_x[j] for j in range(d)]
        for i in range(d):
            b[i] += reward * x[i]
            for j in range(d):
                A[i][j] += x[i] * y[j]
    return _linear_solve(A, b)


def recursive_lstd(transitions, gamma, epsilon):
    if epsilon <= 0:
        raise ValueError("epsilon pozitif olmalı")
    if not transitions:
        return []
    d = len(transitions[0][0])
    B = [[float(i == j) / epsilon for j in range(d)] for i in range(d)]
    b = [0.0] * d
    for x, reward, next_x in transitions:
        y = [x[j] - gamma * next_x[j] for j in range(d)]
        u = [sum(B[i][j] * x[j] for j in range(d)) for i in range(d)]
        v = [sum(y[i] * B[i][j] for i in range(d)) for j in range(d)]
        den = 1 + sum(v[j] * x[j] for j in range(d))
        if abs(den) < 1e-10:
            raise ValueError("Sherman–Morrison paydası sıfıra yakın")
        B = [[B[i][j] - u[i] * v[j] / den for j in range(d)] for i in range(d)]
        b = [b[i] + reward * x[i] for i in range(d)]
    return [sum(B[i][j] * b[j] for j in range(d)) for i in range(d)]
