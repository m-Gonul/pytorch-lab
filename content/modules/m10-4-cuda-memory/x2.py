import numpy as np


def warp_cost(path_ids, costs):
    ...


def kernel_cost(n_threads, branch, costs):
    ...


print(kernel_cost(1024, "thread", np.array([10, 30])))
