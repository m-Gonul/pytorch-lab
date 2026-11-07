import logging
import torch
from labcheck import close, need, ensure, bench, fmt_time

torch._logging.set_logs(dynamo=logging.ERROR)


def check(ns, R):
    def br():
        x = torch.randn(1000)
        a = need(ns, "count_breaks")(ns["branchy"], x)
        b = ns["count_breaks"](need(ns, "branchy_fixed"), x)
        ensure(a >= 1, f"branchy'de en az bir graph break bekleniyordu ({a})")
        ensure(b == 0, f"branchy_fixed'de {b} graph break var")
        return f"branchy: {a} break, branchy_fixed: {b}"
    R.test("graph break'ler sayılıyor ve gideriliyor", br)

    def same():
        for s in (1.0, -1.0):
            x = torch.rand(50) * s
            close(need(ns, "branchy_fixed")(x), ns["branchy"](x), what=f"işaret {s:+.0f}")
    R.test("branchy_fixed = branchy (iki dal)", same)

    def comp():
        torch._dynamo.reset()
        x = torch.randn(3000)
        y_c = need(ns, "compiled_chain")(x)
        close(y_c, need(ns, "chain")(x), atol=1e-5)
    R.test("compiled_chain = chain", comp)
    if "compiled_chain" not in ns:
        return
    x = torch.randn(4_000_000)
    ns["compiled_chain"](x)
    t_e = bench(lambda: ns["chain"](x), repeat=5)
    t_c = bench(lambda: ns["compiled_chain"](x), repeat=5)
    R.table("4M eleman, elementwise zincir (CPU)", ["", "süre"], [["eager", fmt_time(t_e)], ["torch.compile (Inductor)", fmt_time(t_c)]],
            note="Eager her işlem için ayrı bir döngü ve ara tensör; Inductor tek bir C++ döngüsü üretir. Kazanç, işlem bellek-bağımlıyken büyük.")
