import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <vector>

double sum_acc(torch::Tensor x, int64_t nthreads, int64_t k) {
    // TODO
    return 0.0;
}

torch::Tensor hist_copies(torch::Tensor img, int64_t nthreads, int64_t copies) {
    // TODO
    return torch::zeros({256}, torch::kInt64);
}
"""


def build():
    name = "ptl_m105e_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["sum_acc", "hist_copies"], extra_cflags=["-O2"], verbose=False)


ext = build()
print(ext.sum_acc(torch.ones(1000), 2, 4))
