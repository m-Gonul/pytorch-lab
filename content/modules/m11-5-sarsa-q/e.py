import torch
H, W, START, GOAL = 4, 12, 36, 47


def step(s, a):
    ...


def batched_control(n_envs, n_steps, alpha, eps, method="q", seed=0):
    ...


def greedy_returns(Q, max_steps=100):
    ...


Q, rew = batched_control(100, 2000, 0.5, 0.1)
print(greedy_returns(Q)[:10])
