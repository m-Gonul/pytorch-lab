import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <atomic>
#include <vector>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> mandelbrot(int64_t H, int64_t W, int64_t maxit, int64_t nthreads, std::string mode) {
    // TODO: nthreads tane std::thread başlat, satırları mode'a göre dağıt, join et.
    return {torch::zeros({H, W}, torch::kInt32), torch::zeros({nthreads}, torch::kInt64)};
}
"""


def build():
    name = "ptl_m101_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["mandelbrot"], extra_cflags=["-O2"], verbose=False)


ext = build()
img, work = ext.mandelbrot(100, 150, 200, 2, "dynamic")
print(img.float().mean(), work)
