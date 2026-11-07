import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <vector>
#include <atomic>

class Pool {
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i) ws.emplace_back([this] { loop(); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> g(m); stop = true; }
        cv.notify_all();
        for (auto& w : ws) w.join();
    }
    void submit(std::function<void()> f) {
        { std::lock_guard<std::mutex> g(m); q.push(std::move(f)); }
        cv.notify_one();
    }
private:
    void loop() {
        while (true) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [this] { return stop || !q.empty(); });
                if (stop && q.empty()) return;
                f = std::move(q.front());
                q.pop();
            }
            f();
        }
    }
    std::vector<std::thread> ws;
    std::queue<std::function<void()>> q;
    std::mutex m;
    std::condition_variable cv;
    bool stop = false;
};

double pool_sum(torch::Tensor x, int64_t nthreads, int64_t chunk) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat64 && x.is_contiguous(), "float64, bitişik");
    const double* p = x.data_ptr<double>();
    const int64_t N = x.numel();
    const int64_t ntasks = (N + chunk - 1) / chunk;
    std::vector<double> part(ntasks, 0.0);
    std::atomic<int64_t> left(ntasks);
    std::mutex dm;
    std::condition_variable done;
    {
        Pool pool(nthreads);
        for (int64_t k = 0; k < ntasks; ++k) {
            pool.submit([&, k] {
                double s = 0;
                for (int64_t i = k * chunk; i < std::min(N, (k + 1) * chunk); ++i) s += p[i];
                part[k] = s;
                if (left.fetch_sub(1) == 1) { std::lock_guard<std::mutex> g(dm); done.notify_one(); }
            });
        }
        std::unique_lock<std::mutex> lk(dm);
        done.wait(lk, [&] { return left.load() == 0; });
    }
    double s = 0;
    for (double v : part) s += v;
    return s;
}

double spawn_sum(torch::Tensor x, int64_t nthreads) {
    const double* p = x.data_ptr<double>();
    const int64_t N = x.numel();
    std::vector<double> part(nthreads, 0.0);
    std::vector<std::thread> ts;
    for (int64_t t = 0; t < nthreads; ++t)
        ts.emplace_back([&, t] {
            double s = 0;
            for (int64_t i = N * t / nthreads; i < N * (t + 1) / nthreads; ++i) s += p[i];
            part[t] = s;
        });
    for (auto& th : ts) th.join();
    double s = 0;
    for (double v : part) s += v;
    return s;
}
"""


def build():
    name = "ptl_m102e_" + hashlib.md5(SRC.encode()).hexdigest()[:8]
    return load_inline(name=name, cpp_sources=SRC, functions=["pool_sum", "spawn_sum"], extra_cflags=["-O2"], verbose=False)


ext = build()
x = torch.rand(100000, dtype=torch.float64)
print(ext.pool_sum(x, 4, 1000) - float(x.sum()))
