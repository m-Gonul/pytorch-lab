import hashlib
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CPP_SRC = r"""
#include <torch/extension.h>
#include <cmath>

torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor b) {
    // TODO: TORCH_CHECK'ler, sonra y = gelu_tanh(x + b)
    return x;
}
"""

CUDA_SRC = None   # isteğe bağlı: GPU varsa kendi __global__ çekirdeğin


def build():
    name = "ptl_bias_gelu_" + hashlib.md5(CPP_SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=CPP_SRC, functions=["bias_gelu"], verbose=False)


ext = build()
x, b = torch.randn(4, 8), torch.randn(8)
print((ext.bias_gelu(x, b) - F.gelu(x + b, approximate="tanh")).abs().max())
