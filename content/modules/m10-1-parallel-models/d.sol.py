import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <atomic>
#include <vector>
#include <tuple>

static int escape(double cr, double ci, int maxit) {
    double zr = 0, zi = 0;
    int k = 0;
    while (k < maxit && zr * zr + zi * zi <= 4.0) {
        double t = zr * zr - zi * zi + cr;
        zi = 2 * zr * zi + ci;
        zr = t;
        ++k;
    }
    return k;
}

static int64_t do_row(int32_t* out, int64_t r, int64_t H, int64_t W, int maxit) {
    const double ci = 1.5 * r / (H - 1);
    int64_t work = 0;
    for (int64_t c = 0; c < W; ++c) {
        int k = escape(-2.0 + 3.0 * c / (W - 1), ci, maxit);
        out[r * W + c] = k;
        work += k;
    }
    return work;
}

std::tuple<torch::Tensor, torch::Tensor> mandelbrot(int64_t H, int64_t W, int64_t maxit, int64_t nthreads, std::string mode) {
    TORCH_CHECK(nthreads >= 1 && H >= 2 && W >= 2, "geçersiz boyut");
    auto out = torch::zeros({H, W}, torch::kInt32);
    auto work = torch::zeros({nthreads}, torch::kInt64);
    int32_t* p = out.data_ptr<int32_t>();
    int64_t* w = work.data_ptr<int64_t>();
    std::atomic<int64_t> next(0);
    std::vector<std::thread> ts;
    for (int64_t t = 0; t < nthreads; ++t) {
        ts.emplace_back([=, &next]() {
            int64_t acc = 0;
            if (mode == "block") {
                for (int64_t r = H * t / nthreads; r < H * (t + 1) / nthreads; ++r) acc += do_row(p, r, H, W, maxit);
            } else if (mode == "cyclic") {
                for (int64_t r = t; r < H; r += nthreads) acc += do_row(p, r, H, W, maxit);
            } else {
                for (int64_t r = next.fetch_add(1); r < H; r = next.fetch_add(1)) acc += do_row(p, r, H, W, maxit);
            }
            w[t] = acc;
        });
    }
    for (auto& th : ts) th.join();
    return {out, work};
}
"""


def build():
    name = "ptl_m101_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["mandelbrot"], extra_cflags=["-O2"], verbose=False)


ext = build()
img, work = ext.mandelbrot(100, 150, 200, 2, "dynamic")
print(img.float().mean(), work)
