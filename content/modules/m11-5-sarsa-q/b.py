import random
H, W = 4, 12
START, GOAL = 36, 47


def cliff_step(s, a):
    ...


def eps_greedy(Q, s, eps, rng):
    ...


def sarsa(n_eps, alpha=0.5, eps=0.1, seed=0):
    ...


def qlearning(n_eps, alpha=0.5, eps=0.1, seed=0):
    ...


def greedy_return(Q, max_steps=100):
    ...


print(greedy_return(qlearning(300)[0]))
