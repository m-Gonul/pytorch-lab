import os
import torch
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R_):
    ext = need(ns, "build")()
    P = max(2, min(8, os.cpu_count() or 2))

    def t():
        g = torch.Generator().manual_seed(0)
        x = torch.rand(10_000_000, generator=g, dtype=torch.float64)
        ref = float(x.sum())
        for ch in (1000, 123457, 10_000_000):
            close(ext.pool_sum(x, P, ch), ref, rtol=1e-12, what=f"havuz toplamı (chunk {ch})")
        close(ext.spawn_sum(x, P), ref, rtol=1e-12, what="spawn toplamı")
        rows = []
        for ch in (100, 1000, 10_000, 100_000, 1_000_000):
            tt = bench(lambda: ext.pool_sum(x, P, ch), repeat=3, number=1)
            rows.append([f"havuz, chunk {ch:,}".replace(",", " "), f"{(x.numel() + ch - 1) // ch:,}".replace(",", " "), fmt_time(tt), fmt_time(tt / ((x.numel() + ch - 1) // ch))])
        rows.append([f"{P} thread aç-kapat (eşit bölme)", P, fmt_time(bench(lambda: ext.spawn_sum(x, P), repeat=3, number=1)), ""])
        old = torch.get_num_threads()
        for nt in sorted({1, P}):
            torch.set_num_threads(nt)
            rows.append([f"torch.sum, {nt} iç thread", "", fmt_time(bench(lambda: x.sum(), repeat=5, number=1)), ""])
        torch.set_num_threads(old)
        R_.table(f"10 M float64 toplamı, {P} işçi", ["yöntem", "görev", "süre", "görev başına"], rows,
                 note="Çok küçük görevlerde süreyi kuyruk kilidi, std::function tahsisi ve uyandırmalar belirler; çok büyük görevlerde paralellik kaybolur (tek görev = tek thread). Pratik kural: görev başına en az ~10–100 µs iş.")
    R_.test("thread havuzu ve ayrıntı düzeyi", t)
