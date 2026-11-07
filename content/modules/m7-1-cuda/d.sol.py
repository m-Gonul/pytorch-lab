import hashlib
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CPP_SRC = r"""
#include <torch/extension.h>
#include <cmath>

torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor b) {
    TORCH_CHECK(x.dim() == 2 && b.dim() == 1 && x.size(1) == b.size(0), "x (n, d), b (d,) olmalı");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "fp32 bekleniyor");
    TORCH_CHECK(x.is_contiguous() && b.is_contiguous(), "bitişik tensör bekleniyor");
    auto y = torch::empty_like(x);
    const int64_t n = x.size(0), d = x.size(1);
    const float* xp = x.data_ptr<float>();
    const float* bp = b.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            const float v = xp[i * d + j] + bp[j];
            yp[i * d + j] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
        }
    }
    return y;
}
"""

CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void bias_gelu_kernel(const float* x, const float* b, float* y, int64_t n, int64_t d) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * d) {
        const float v = x[i] + b[i % d];
        y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

torch::Tensor bias_gelu_cuda(torch::Tensor x, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda() && b.is_cuda(), "CUDA tensörü bekleniyor");
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32, "fp32, bitişik");
    auto y = torch::empty_like(x);
    const int64_t N = x.numel();
    const int threads = 256;
    const int64_t blocks = (N + threads - 1) / threads;
    bias_gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}
"""   # yalnız GPU'da derlenir


def build():
    name = "ptl_bias_gelu_" + hashlib.md5(CPP_SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=CPP_SRC, functions=["bias_gelu"], verbose=False)


ext = build()
x, b = torch.randn(4, 8), torch.randn(8)
print((ext.bias_gelu(x, b) - F.gelu(x + b, approximate="tanh")).abs().max())
