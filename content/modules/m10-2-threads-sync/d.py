import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>
#include <tuple>

torch::Tensor histogram(torch::Tensor img, int64_t nthreads, std::string mode) {
    // TODO
    return torch::zeros({256}, torch::kInt64);
}

std::tuple<double, torch::Tensor> false_sharing(int64_t nthreads, int64_t iters, bool padded) {
    // TODO
    return {0.0, torch::zeros({nthreads}, torch::kInt64)};
}

std::tuple<int64_t, int64_t> bounded_buffer(int64_t n_items, int64_t cap, int64_t n_prod, int64_t n_cons) {
    // TODO
    return {0, 0};
}
"""


def build():
    name = "ptl_m102_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["histogram", "false_sharing", "bounded_buffer"], extra_cflags=["-O2"], verbose=False)


ext = build()
img = torch.randint(0, 256, (256, 256), dtype=torch.uint8)
print((ext.histogram(img, 4, "private") - torch.bincount(img.flatten(), minlength=256)).abs().sum())
