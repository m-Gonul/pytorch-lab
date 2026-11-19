def gain_step(rho, delta, n, beta, trace, debias=True):
    # TODO: raw delta içinde n*rho var; ortalama hedefi için delta/n
    raise NotImplementedError


def differential_n_sarsa(steps=5000, n=2, alpha=.01, beta=.01, epsilon=.1,
                         seed=0, debias=True, initial_gain=0.0):
    # TODO: n+1 halka tamponu, ortak güncel rho, sürekli n-adım SARSA
    raise NotImplementedError
