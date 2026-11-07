import numpy as np


def warp_cost(path_ids, costs):
    costs = np.asarray(costs)
    used = (np.asarray(path_ids)[:, :, None] == np.arange(len(costs))).any(1)
    return used.astype(float) @ costs


def kernel_cost(n_threads, branch, costs):
    tid = np.arange(n_threads)
    if branch == "thread":
        p = tid % 2
    elif branch == "warp":
        p = (tid // 32) % 2
    else:
        p = np.zeros_like(tid)
    return float(warp_cost(p.reshape(-1, 32), costs).sum())


print(kernel_cost(1024, "thread", np.array([10, 30])))
