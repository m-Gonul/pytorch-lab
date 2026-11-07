import os
import time
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
from labcheck import close, need, ensure, bench, fmt_time

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <atomic>
#include <vector>
#include <tuple>
static int esc(double cr, double ci, int m) { double zr = 0, zi = 0; int k = 0; while (k < m && zr * zr + zi * zi <= 4.0) { double t = zr * zr - zi * zi + cr; zi = 2 * zr * zi + ci; zr = t; ++k; } return k; }
std::tuple<torch::Tensor, torch::Tensor> mandelbrot(int64_t H, int64_t W, int64_t maxit, int64_t nt, std::string mode) {
    auto out = torch::zeros({H, W}, torch::kInt32); auto work = torch::zeros({nt}, torch::kInt64);
    int32_t* p = out.data_ptr<int32_t>(); std::atomic<int64_t> nx(0); std::vector<std::thread> ts;
    for (int64_t t = 0; t < nt; ++t) ts.emplace_back([=, &nx]() { for (int64_t r = nx.fetch_add(1); r < H; r = nx.fetch_add(1)) { double ci = 1.5 * r / (H - 1); for (int64_t c = 0; c < W; ++c) p[r * W + c] = esc(-2.0 + 3.0 * c / (W - 1), ci, maxit); } });
    for (auto& th : ts) th.join();
    return {out, work};
}
"""


def py_mandel(H, W, maxit):
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        ci = 1.5 * r / (H - 1)
        for c in range(W):
            cr = -2.0 + 3.0 * c / (W - 1)
            zr = zi = 0.0
            k = 0
            while k < maxit and zr * zr + zi * zi <= 4.0:
                zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
                k += 1
            out[r][c] = k
    return out


def check(ns, R_):
    ext = load_inline(name="ptl_m101e_" + hashlib.md5(SRC.encode()).hexdigest()[:8], cpp_sources=SRC, functions=["mandelbrot"], extra_cflags=["-O2"], verbose=False)

    def t():
        H, W, it = 300, 450, 500
        mt = need(ns, "mandel_torch")(H, W, it)
        mc, _ = ext.mandelbrot(H, W, it, 1, "dynamic")
        ensure(float((mt == mc).double().mean()) > 0.999, "torch sürümü C++ ile aynı")
        ncpu = os.cpu_count() or 2
        threads = sorted({1, 2, max(2, ncpu // 2), ncpu})
        T, S = need(ns, "scaling")(ext, 600, 900, 1000, threads)
        s, T1 = need(ns, "fit_amdahl")(threads, T)
        t_torch = bench(lambda: ns["mandel_torch"](H, W, it), repeat=1, number=1)
        t_cpp = bench(lambda: ext.mandelbrot(H, W, it, ncpu, "dynamic"), repeat=3, number=1)
        t_py = bench(lambda: py_mandel(30, 45, it), repeat=1, number=1) * 100
        rows = [[f"C++ dinamik, {p} thread (600×900, maxit 1000)", fmt_time(T[i]), f"{S[i]:.2f}×"] for i, p in enumerate(threads)]
        rows += [[f"saf Python ({H}×{W}, tahmini)", fmt_time(t_py), ""], [f"PyTorch vektörel ({H}×{W})", fmt_time(t_torch), ""], [f"C++ {ncpu} thread ({H}×{W})", fmt_time(t_cpp), ""]]
        R_.table(f"Mandelbrot ölçekleme ({ncpu} mantıksal çekirdek)", ["uygulama", "süre", "hızlanma"], rows,
                 note=f"Amdahl uydurması: seri oran ≈ {max(s, 0):.3f}. Vektörel PyTorch, kaçmış pikselleri de her iterasyonda hesaplar (maske): iş verimsiz ama Python döngüsü yok. C++'ta her piksel yalnız kendi iterasyonu kadar çalışır.")
        ensure(S[-1] > 1.2 or ncpu < 2, "çok thread hızlandırmalı")
    R_.test("ölçekleme", t)
