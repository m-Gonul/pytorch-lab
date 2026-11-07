import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <vector>
#include <algorithm>

template <class F>
static void parallel(int64_t nthreads, F f) {
    if (nthreads <= 1) { f(0); return; }
    std::vector<std::thread> ts;
    for (int64_t t = 0; t < nthreads; ++t) ts.emplace_back(f, t);
    for (auto& th : ts) th.join();
}

torch::Tensor integral_array(torch::Tensor x, int64_t nthreads) {
    TORCH_CHECK(x.scalar_type() == torch::kInt32 && x.is_contiguous(), "int32, bitişik");
    const int64_t n = x.numel();
    const int32_t* a = x.data_ptr<int32_t>();
    auto out = torch::empty({n}, torch::kInt64);
    int64_t* s = out.data_ptr<int64_t>();
    std::vector<int64_t> tot(nthreads, 0);
    parallel(nthreads, [&](int64_t t) {
        int64_t acc = 0;
        for (int64_t i = n * t / nthreads; i < n * (t + 1) / nthreads; ++i) { acc += a[i]; s[i] = acc; }
        tot[t] = acc;
    });
    std::vector<int64_t> off(nthreads, 0);
    for (int64_t t = 1; t < nthreads; ++t) off[t] = off[t - 1] + tot[t - 1];
    parallel(nthreads, [&](int64_t t) {
        if (t == 0) return;
        for (int64_t i = n * t / nthreads; i < n * (t + 1) / nthreads; ++i) s[i] += off[t];
    });
    return out;
}

torch::Tensor integral_image(torch::Tensor img, int64_t nthreads) {
    TORCH_CHECK(img.scalar_type() == torch::kUInt8 && img.dim() == 2 && img.is_contiguous(), "uint8 (H, W)");
    const int64_t H = img.size(0), W = img.size(1);
    const uint8_t* f = img.data_ptr<uint8_t>();
    auto out = torch::empty({H, W}, torch::kInt64);
    int64_t* I = out.data_ptr<int64_t>();
    parallel(nthreads, [&](int64_t t) {
        for (int64_t y = H * t / nthreads; y < H * (t + 1) / nthreads; ++y) {
            int64_t acc = 0;
            for (int64_t x = 0; x < W; ++x) { acc += f[y * W + x]; I[y * W + x] = acc; }
        }
    });
    parallel(nthreads, [&](int64_t t) {
        const int64_t x0 = W * t / nthreads, x1 = W * (t + 1) / nthreads;
        for (int64_t y = 1; y < H; ++y)
            for (int64_t x = x0; x < x1; ++x) I[y * W + x] += I[(y - 1) * W + x];
    });
    return out;
}

torch::Tensor conv2d(torch::Tensor img, torch::Tensor k, int64_t nthreads) {
    TORCH_CHECK(img.scalar_type() == torch::kUInt8 && img.dim() == 2 && img.is_contiguous(), "uint8 (H, W)");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32 && k.dim() == 2 && k.size(0) == k.size(1) && k.size(0) % 2 == 1, "fp32 tek kenarlı kare filtre");
    k = k.contiguous();
    const int64_t H = img.size(0), W = img.size(1), K = k.size(0), r = K / 2;
    const uint8_t* f = img.data_ptr<uint8_t>();
    const float* w = k.data_ptr<float>();
    auto out = torch::empty({H, W}, torch::kUInt8);
    uint8_t* o = out.data_ptr<uint8_t>();
    parallel(nthreads, [&](int64_t t) {
        for (int64_t y = H * t / nthreads; y < H * (t + 1) / nthreads; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float acc = 0.f;
                for (int64_t u = 0; u < K; ++u) {
                    const int64_t yy = y + u - r;
                    if (yy < 0 || yy >= H) continue;
                    const uint8_t* row = f + yy * W;
                    const float* wr = w + u * K;
                    for (int64_t v = 0; v < K; ++v) {
                        const int64_t xx = x + v - r;
                        if (xx >= 0 && xx < W) acc += wr[v] * row[xx];
                    }
                }
                int q = (int)(acc + 0.5f);
                o[y * W + x] = (uint8_t)std::min(255, std::max(0, q));
            }
        }
    });
    return out;
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B, int64_t nthreads, std::string mode) {
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32 && A.size(1) == B.size(0), "fp32, uyumlu boyut");
    A = A.contiguous();
    B = B.contiguous();
    const int64_t n = A.size(0), m = A.size(1), q = B.size(1);
    auto C = torch::zeros({n, q}, torch::kFloat32);
    const float* a = A.data_ptr<float>();
    const float* b = B.data_ptr<float>();
    float* c = C.data_ptr<float>();
    torch::Tensor Bt;
    const float* bt = nullptr;
    if (mode == "bt") { Bt = B.t().contiguous(); bt = Bt.data_ptr<float>(); }
    const int64_t T = 64;
    parallel(nthreads, [&](int64_t t) {
        const int64_t i0 = n * t / nthreads, i1 = n * (t + 1) / nthreads;
        if (mode == "ijk") {
            for (int64_t i = i0; i < i1; ++i)
                for (int64_t j = 0; j < q; ++j) {
                    float s = 0.f;
                    for (int64_t kk = 0; kk < m; ++kk) s += a[i * m + kk] * b[kk * q + j];
                    c[i * q + j] = s;
                }
        } else if (mode == "bt") {
            for (int64_t i = i0; i < i1; ++i)
                for (int64_t j = 0; j < q; ++j) {
                    float s = 0.f;
                    for (int64_t kk = 0; kk < m; ++kk) s += a[i * m + kk] * bt[j * m + kk];
                    c[i * q + j] = s;
                }
        } else if (mode == "ikj") {
            for (int64_t i = i0; i < i1; ++i)
                for (int64_t kk = 0; kk < m; ++kk) {
                    const float av = a[i * m + kk];
                    for (int64_t j = 0; j < q; ++j) c[i * q + j] += av * b[kk * q + j];
                }
        } else {
            for (int64_t ii = i0; ii < i1; ii += T)
                for (int64_t kk0 = 0; kk0 < m; kk0 += T)
                    for (int64_t jj = 0; jj < q; jj += T)
                        for (int64_t i = ii; i < std::min(ii + T, i1); ++i)
                            for (int64_t kk = kk0; kk < std::min(kk0 + T, m); ++kk) {
                                const float av = a[i * m + kk];
                                for (int64_t j = jj; j < std::min(jj + T, q); ++j) c[i * q + j] += av * b[kk * q + j];
                            }
        }
    });
    return C;
}
"""


def build():
    name = "ptl_m103_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["integral_array", "integral_image", "conv2d", "matmul"], extra_cflags=["-O3"], verbose=False)


ext = build()
x = torch.randint(-5, 5, (4096,), dtype=torch.int32)
print((ext.integral_array(x, 4) - torch.cumsum(x.long(), 0)).abs().max())
