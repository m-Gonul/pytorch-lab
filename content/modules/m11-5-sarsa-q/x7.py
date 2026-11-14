def q_target(states, actions, rewards, Q, pi, behavior, tau, n, kind="is", sigma=0.5, gamma=1.0):
    # TODO: doğru ufuk, terminal ve önem oranı.
    return 0.0, 1.0


def learn_q_chain(episodes=10000, n=4, alpha=0.05, kind="sigma", sigma=0.5, seed=0):
    # TODO: verilen dört durumlu zincirde Q tahmini.
    return [[0.0, 0.0] for _ in range(5)]
