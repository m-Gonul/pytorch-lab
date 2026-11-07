import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>
#include <tuple>

torch::Tensor histogram(torch::Tensor img, int64_t nthreads, std::string mode) {
    TORCH_CHECK(img.scalar_type() == torch::kUInt8 && img.is_contiguous(), "uint8, bitişik");
    const uint8_t* p = img.data_ptr<uint8_t>();
    const int64_t N = img.numel();
    auto out = torch::zeros({256}, torch::kInt64);
    int64_t* h = out.data_ptr<int64_t>();
    if (mode == "single") {
        for (int64_t i = 0; i < N; ++i) h[p[i]]++;
        return out;
    }
    std::vector<std::atomic<int64_t>> ab(256);
    for (auto& a : ab) a.store(0);
    std::mutex m;
    std::vector<std::thread> ts;
    for (int64_t t = 0; t < nthreads; ++t) {
        ts.emplace_back([&, t]() {
            const int64_t lo = N * t / nthreads, hi = N * (t + 1) / nthreads;
            if (mode == "atomic") {
                for (int64_t i = lo; i < hi; ++i) ab[p[i]].fetch_add(1, std::memory_order_relaxed);
            } else if (mode == "racy") {
                for (int64_t i = lo; i < hi; ++i) {
                    int64_t v = ab[p[i]].load(std::memory_order_relaxed);
                    ab[p[i]].store(v + 1, std::memory_order_relaxed);
                }
            } else if (mode == "mutex") {
                for (int64_t i = lo; i < hi; ++i) {
                    std::lock_guard<std::mutex> g(m);
                    h[p[i]]++;
                }
            } else {
                int64_t local[256] = {0};
                for (int64_t i = lo; i < hi; ++i) local[p[i]]++;
                std::lock_guard<std::mutex> g(m);
                for (int b = 0; b < 256; ++b) h[b] += local[b];
            }
        });
    }
    for (auto& th : ts) th.join();
    if (mode == "atomic" || mode == "racy")
        for (int b = 0; b < 256; ++b) h[b] = ab[b].load();
    return out;
}

struct alignas(64) Padded { std::atomic<int64_t> v; };

std::tuple<double, torch::Tensor> false_sharing(int64_t nthreads, int64_t iters, bool padded) {
    std::vector<std::atomic<int64_t>> plain(nthreads);
    std::vector<Padded> pad(nthreads);
    for (int64_t i = 0; i < nthreads; ++i) { plain[i].store(0); pad[i].v.store(0); }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int64_t t = 0; t < nthreads; ++t) {
        ts.emplace_back([&, t]() {
            std::atomic<int64_t>& c = padded ? pad[t].v : plain[t];
            for (int64_t k = 0; k < iters; ++k) c.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : ts) th.join();
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto out = torch::zeros({nthreads}, torch::kInt64);
    for (int64_t i = 0; i < nthreads; ++i) out[i] = padded ? pad[i].v.load() : plain[i].load();
    return {dt, out};
}

std::tuple<int64_t, int64_t> bounded_buffer(int64_t n_items, int64_t cap, int64_t n_prod, int64_t n_cons) {
    std::queue<int64_t> q;
    std::mutex m;
    std::condition_variable not_full, not_empty;
    std::atomic<int64_t> total(0);
    int64_t max_occ = 0;
    std::vector<std::thread> prod, cons;
    for (int64_t c = 0; c < n_cons; ++c) {
        cons.emplace_back([&]() {
            while (true) {
                std::unique_lock<std::mutex> lk(m);
                not_empty.wait(lk, [&] { return !q.empty(); });
                int64_t v = q.front();
                q.pop();
                lk.unlock();
                not_full.notify_one();
                if (v < 0) return;
                total.fetch_add(v);
            }
        });
    }
    for (int64_t p = 0; p < n_prod; ++p) {
        prod.emplace_back([&, p]() {
            for (int64_t v = 1 + p; v <= n_items; v += n_prod) {
                std::unique_lock<std::mutex> lk(m);
                not_full.wait(lk, [&] { return (int64_t)q.size() < cap; });
                q.push(v);
                if ((int64_t)q.size() > max_occ) max_occ = q.size();
                lk.unlock();
                not_empty.notify_one();
            }
        });
    }
    for (auto& th : prod) th.join();
    for (int64_t c = 0; c < n_cons; ++c) {
        std::unique_lock<std::mutex> lk(m);
        not_full.wait(lk, [&] { return (int64_t)q.size() < cap; });
        q.push(-1);
        lk.unlock();
        not_empty.notify_one();
    }
    for (auto& th : cons) th.join();
    return {total.load(), max_occ};
}
"""


def build():
    name = "ptl_m102_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["histogram", "false_sharing", "bounded_buffer"], extra_cflags=["-O2"], verbose=False)


ext = build()
img = torch.randint(0, 256, (256, 256), dtype=torch.uint8)
print((ext.histogram(img, 4, "private") - torch.bincount(img.flatten(), minlength=256)).abs().sum())
