import math
import numpy as np


def make_logs(n, seed=0):
    ...


def replay_linucb(X, A, R, alpha=1.0):
    ...


def online_linucb(theta, T, seed=0, alpha=1.0):
    ...


X, A, R, th = make_logs(5000)
print(replay_linucb(X, A, R))
