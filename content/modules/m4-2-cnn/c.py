import numpy as np


def grad_norms(X, Ws, residual):
    """→ [‖g_0‖, …, ‖g_L‖]. Yalnız katman döngüleri."""
    ...


rng = np.random.default_rng(0)
D, L = 64, 20
X = rng.standard_normal((32, D))
Ws = [rng.standard_normal((D, D)) * 0.5 / np.sqrt(D) for _ in range(L)]
print(grad_norms(X, Ws, False)[0], grad_norms(X, Ws, True)[0])
