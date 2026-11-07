import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <vector>
#include <atomic>

double pool_sum(torch::Tensor x, int64_t nthreads, int64_t chunk) {
    // TODO: thread havuzu
    return 0.0;
}

double spawn_sum(torch::Tensor x, int64_t nthreads) {
    // TODO
    return 0.0;
}
"""


def build():
    name = "ptl_m102e_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["pool_sum", "spawn_sum"], extra_cflags=["-O2"], verbose=False)


ext = build()
x = torch.rand(100000, dtype=torch.float64)
print(ext.pool_sum(x, 4, 1000) - float(x.sum()))
