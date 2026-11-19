def differential_torch_step(Q, rho, state, action, reward, next_state, next_action,
                            alpha=0.01, beta=0.01):
    # TODO: hedefi detach et; yarım kare kayıptan yalnız mevcut Q için gradyan
    raise NotImplementedError


def torch_sarsa(steps=5000, alpha=0.01, beta=0.01, epsilon=0.1, seed=0):
    # TODO: sürekli on-policy kontrol, terminal/reset/indirim yok
    raise NotImplementedError
