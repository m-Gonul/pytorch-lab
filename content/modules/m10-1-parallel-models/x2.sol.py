import heapq


def simulate(stages, replicas, n_items):
    free = [[0.0] * r for r in replicas]
    t_done = 0.0
    for _ in range(n_items):
        t = 0.0
        for s, d in enumerate(stages):
            k = min(range(replicas[s]), key=lambda i: free[s][i])
            t = max(t, free[s][k]) + d
            free[s][k] = t
        t_done = t
    return t_done


def throughput(stages, replicas):
    return 1 / max(t / r for t, r in zip(stages, replicas))


print(simulate([2, 3, 5, 2, 3], [1, 1, 2, 1, 1], 100))
