import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <vector>
#include <algorithm>

torch::Tensor integral_array(torch::Tensor x, int64_t nthreads) {
    // TODO
    return torch::zeros({x.numel()}, torch::kInt64);
}

torch::Tensor integral_image(torch::Tensor img, int64_t nthreads) {
    // TODO
    return torch::zeros(img.sizes(), torch::kInt64);
}

torch::Tensor conv2d(torch::Tensor img, torch::Tensor k, int64_t nthreads) {
    // TODO
    return torch::zeros_like(img);
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B, int64_t nthreads, std::string mode) {
    // TODO
    return torch::zeros({A.size(0), B.size(1)}, torch::kFloat32);
}
"""


def build():
    name = "ptl_m103_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["integral_array", "integral_image", "conv2d", "matmul"], extra_cflags=["-O3"], verbose=False)


ext = build()
x = torch.randint(-5, 5, (4096,), dtype=torch.int32)
print((ext.integral_array(x, 4) - torch.cumsum(x.long(), 0)).abs().max())
