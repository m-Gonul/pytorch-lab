import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>

torch::Tensor matmul_ijk(torch::Tensor A, torch::Tensor B) {
    // TODO
    return A;
}

torch::Tensor matmul_ikj(torch::Tensor A, torch::Tensor B) {
    // TODO
    return A;
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B, int64_t T) {
    // TODO
    return A;
}
"""


def build():
    name = "ptl_mm_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["matmul_ijk", "matmul_ikj", "matmul_tiled"],
                       extra_cflags=["-O3"], verbose=False)


ext = build()
A, B = torch.randn(64, 64), torch.randn(64, 64)
print((ext.matmul_ikj(A, B) - A @ B).abs().max())
