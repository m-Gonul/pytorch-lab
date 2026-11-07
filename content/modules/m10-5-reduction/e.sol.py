import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <vector>

double sum_acc(torch::Tensor x, int64_t nthreads, int64_t k) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "float32, bitişik");
    TORCH_CHECK(k >= 1 && k <= 16, "1 <= k <= 16");
    const float* p = x.data_ptr<float>();
    const int64_t n = x.numel();
    std::vector<double> part(nthreads, 0.0);
    std::vector<std::thread> ts;
    for (int64_t t = 0; t < nthreads; ++t) {
        ts.emplace_back([&, t] {
            const int64_t lo = n * t / nthreads, hi = n * (t + 1) / nthreads;
            float acc[16] = {0};
            int64_t i = lo;
            for (; i + k <= hi; i += k)
                for (int64_t j = 0; j < k; ++j) acc[j] += p[i + j];
            for (; i < hi; ++i) acc[0] += p[i];
            double s = 0;
            for (int64_t j = 0; j < k; ++j) s += acc[j];
            part[t] = s;
        });
    }
    for (auto& th : ts) th.join();
    double s = 0;
    for (double v : part) s += v;
    return s;
}

torch::Tensor hist_copies(torch::Tensor img, int64_t nthreads, int64_t copies) {
    TORCH_CHECK(img.scalar_type() == torch::kUInt8 && img.is_contiguous(), "uint8, bitişik");
    const uint8_t* p = img.data_ptr<uint8_t>();
    const int64_t n = img.numel();
    std::vector<std::vector<int64_t>> priv(nthreads, std::vector<int64_t>(256 * copies, 0));
    std::vector<std::thread> ts;
    for (int64_t t = 0; t < nthreads; ++t) {
        ts.emplace_back([&, t] {
            int64_t* h = priv[t].data();
            const int64_t lo = n * t / nthreads, hi = n * (t + 1) / nthreads;
            if (copies == 1) {
                for (int64_t i = lo; i < hi; ++i) h[p[i]]++;
            } else {
                int64_t i = lo;
                for (; i + copies <= hi; i += copies)
                    for (int64_t c = 0; c < copies; ++c) h[c * 256 + p[i + c]]++;
                for (; i < hi; ++i) h[p[i]]++;
            }
        });
    }
    for (auto& th : ts) th.join();
    auto out = torch::zeros({256}, torch::kInt64);
    int64_t* o = out.data_ptr<int64_t>();
    for (auto& v : priv)
        for (int64_t c = 0; c < copies; ++c)
            for (int b = 0; b < 256; ++b) o[b] += v[c * 256 + b];
    return out;
}
"""


def build():
    name = "ptl_m105e_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["sum_acc", "hist_copies"], extra_cflags=["-O2"], verbose=False)


ext = build()
print(ext.sum_acc(torch.ones(1000), 2, 4))
