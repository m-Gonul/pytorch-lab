import numpy as np

WIND = np.array([0,0,0,1,1,1,2,2,1,0])
MOVES = [(-1,0),(0,1),(1,0),(0,-1),(-1,-1),(-1,1),(1,1),(1,-1),(0,0)]
START, GOAL = 30, 37


def windy_step(s,a,stochastic=False,rng=None):
    # TODO: kaynak sütunun rüzgârı, pozitif sütunlarda üç olası değişim.
    raise NotImplementedError


def optimal_windy(actions=8,stochastic=False):
    # TODO: bağımsız Bellman referansı; hedef değeri0.
    raise NotImplementedError


def sarsa_windy(steps=8000,actions=8,stochastic=False,alpha=.5,epsilon=.1,seed=1):
    # TODO: A' güncellemeden önce seçilsin; bitmeyen son parça terminal sayılmasın.
    raise NotImplementedError


def trajectory(policy,stochastic=False,seed=1,budget=1000):
    # TODO: yörünge ve gerçek sonlanma bayrağı.
    raise NotImplementedError
