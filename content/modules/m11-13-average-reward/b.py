def access_step(free, priority, action, rng, p=0.06, release_new=True):
    # TODO: kabul/ödül, meşgul sunucuların bağımsız bırakılması, yeni öncelik
    raise NotImplementedError


def differential_sarsa(steps=20000, alpha=0.01, beta=0.01, epsilon=0.1,
                       seed=0, p=0.06, release_new=True):
    # TODO: sonraki eylemi güncellemeden önce seç; aynı delta ile rho ve Q
    raise NotImplementedError
