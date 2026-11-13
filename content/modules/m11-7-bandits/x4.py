import numpy as np


def bandit_sweep(n_runs, n_steps, configs, seed=0, drift=0.01, noise=1, means=None):
    # Gerçek değerleri yöntemler arasında paylaş; Q/H ve sayımları ayrı tut.
    # Son n_steps - n_steps//2 ödülün ortalamasını döndür.
    raise NotImplementedError
