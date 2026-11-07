import heapq
import random

H, W = 6, 9
WALLS = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
START, GOAL = (2, 0), (0, 8)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def maze_step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= r < H and 0 <= c < W) or (r, c) in WALLS:
        r, c = s
    return (r, c), (1.0 if (r, c) == GOAL else 0.0), (r, c) == GOAL


def greedy_len(Q):
    s = START
    for k in range(1, 60):
        qs = Q.get(s, [0.0] * 4)
        s, _, done = maze_step(s, qs.index(max(qs)))
        if done:
            return k
    return 999


def prioritized(n_plan, theta=1e-4, seed=0, max_steps=20000):
    ...


def dyna_updates(n_plan, seed=0, max_steps=20000):
    ...


print(prioritized(5), dyna_updates(5))
