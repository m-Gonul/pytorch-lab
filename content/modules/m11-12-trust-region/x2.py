import numpy as np


def surrogates(t_grid, theta0, d, A, eps=0.2, beta=1.0):
    ...


t = np.linspace(-3, 3, 7)
print(surrogates(t, np.zeros(3), np.array([1.0, -1.0, 0.0]), np.array([1.0, -0.5, 0.0])))
