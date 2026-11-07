import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <algorithm>

static void check(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0), "boyutlar uyuşmuyor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "fp32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "bitişik");
}

torch::Tensor matmul_ijk(torch::Tensor A, torch::Tensor B) {
    check(A, B);
    const int64_t n = A.size(0), K = A.size(1), m = B.size(1);
    auto C = torch::zeros({n, m}, A.options());
    const float *a = A.data_ptr<float>(), *b = B.data_ptr<float>();
    float* c = C.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) {
            float s = 0.f;
            for (int64_t k = 0; k < K; ++k) s += a[i * K + k] * b[k * m + j];
            c[i * m + j] = s;
        }
    return C;
}

torch::Tensor matmul_ikj(torch::Tensor A, torch::Tensor B) {
    check(A, B);
    const int64_t n = A.size(0), K = A.size(1), m = B.size(1);
    auto C = torch::zeros({n, m}, A.options());
    const float *a = A.data_ptr<float>(), *b = B.data_ptr<float>();
    float* c = C.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < K; ++k) {
            const float aik = a[i * K + k];
            for (int64_t j = 0; j < m; ++j) c[i * m + j] += aik * b[k * m + j];
        }
    return C;
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B, int64_t T) {
    check(A, B);
    const int64_t n = A.size(0), K = A.size(1), m = B.size(1);
    auto C = torch::zeros({n, m}, A.options());
    const float *a = A.data_ptr<float>(), *b = B.data_ptr<float>();
    float* c = C.data_ptr<float>();
    for (int64_t i0 = 0; i0 < n; i0 += T)
        for (int64_t k0 = 0; k0 < K; k0 += T)
            for (int64_t j0 = 0; j0 < m; j0 += T)
                for (int64_t i = i0; i < std::min(i0 + T, n); ++i)
                    for (int64_t k = k0; k < std::min(k0 + T, K); ++k) {
                        const float aik = a[i * K + k];
                        for (int64_t j = j0; j < std::min(j0 + T, m); ++j) c[i * m + j] += aik * b[k * m + j];
                    }
    return C;
}
"""


def build():
    name = "ptl_mm_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["matmul_ijk", "matmul_ikj", "matmul_tiled"],
                       extra_cflags=["-O3"], verbose=False)


ext = build()
A, B = torch.randn(64, 64), torch.randn(64, 64)
print((ext.matmul_ikj(A, B) - A @ B).abs().max())
