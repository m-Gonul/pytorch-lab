import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
from labcheck import close, need, ensure, bench, fmt_time


def saved_bytes(fn, x):
    tot = [0]

    def pack(t):
        tot[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(x)
    return tot[0]


def check(ns, R):
    f = need(ns, "mem_eff_silu")

    def gc():
        x = torch.randn(20, dtype=torch.float64, requires_grad=True)
        ensure(gradcheck(f, (x,)), "gradcheck")
        close(f(x), F.silu(x), what="ileri")
    R.test("gradcheck ve ileri = F.silu", gc)

    def mem():
        x = torch.randn(1024, 1024, requires_grad=True)
        naive = saved_bytes(lambda t: t * torch.sigmoid(t), x)
        mine = saved_bytes(f, x)
        ref = saved_bytes(F.silu, x)
        ensure(mine == x.numel() * 4, f"yalnız x saklanmalı: {mine:,} bayt")
        R.table("Saklanan bayt (1024 × 1024 fp32)", ["yöntem", "bayt"],
                [["x * sigmoid(x)", f"{naive:,}"], ["F.silu (yerleşik)", f"{ref:,}"], ["senin MemEffSiLU", f"{mine:,}"]])
    R.test("yalnız girdi saklanıyor", mem)
    x = torch.randn(2048, 2048, requires_grad=True)

    def run(fn):
        x.grad = None
        fn(x).sum().backward()
    t_naive = bench(lambda: run(lambda t: t * torch.sigmoid(t)), repeat=3)
    t_mine = bench(lambda: run(f), repeat=3)
    t_ref = bench(lambda: run(F.silu), repeat=3)
    R.table("İleri + geri, 2048 × 2048", ["yöntem", "süre"],
            [["x * sigmoid(x)", fmt_time(t_naive)], ["senin MemEffSiLU", fmt_time(t_mine)], ["F.silu", fmt_time(t_ref)]],
            note="Yeniden hesaplama (sigmoid'i geri geçişte tekrar) az bellek karşılığında biraz ek iş demek; elementwise işlemlerde bu genelde ucuz.")
