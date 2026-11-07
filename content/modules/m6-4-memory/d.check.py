import torch
from labcheck import close, need, ensure


def check(ns, R):
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Tanh()) for _ in range(16))
    x = torch.randn(64, 256, requires_grad=True)
    P = 16 * (256 * 256 + 256) * 4

    def pb():
        ensure(need(ns, "param_bytes")(blocks) == P, f"{ns['param_bytes'](blocks)} ≠ {P}")
    R.test("param_bytes", pb)

    def ob():
        opt = torch.optim.Adam(blocks.parameters())
        ensure(need(ns, "optimizer_state_bytes")(opt) == 0, "ilk adımdan önce durum boş")
        for p in blocks.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        b = ns["optimizer_state_bytes"](opt)
        n_t = len(list(blocks.parameters()))
        ensure(2 * P <= b <= 2 * P + 8 * n_t, f"Adam durumu {b:,} bayt; ≈ 2 × {P:,} bekleniyordu")
        return f"parametre {P:,} bayt, Adam durumu {b:,} bayt (≈ 2×)"
    R.test("optimizer_state_bytes (Adam ≈ 2× parametre)", ob)

    def sb():
        def full():
            h = x
            for b in blocks:
                h = b(h)
            return h
        out, full_b = need(ns, "saved_bytes")(full)
        ensure(full_b == 16 * (65536 + 262144 + 65536), f"checkpoint'siz {full_b:,} bayt")
        rows = [["yok", f"{full_b:,}"]]
        ref_grad = torch.autograd.grad(out.sum(), [x] + list(blocks.parameters()))
        for k in (2, 4, 8):
            o, b = ns["saved_bytes"](lambda: need(ns, "run_checkpointed")(blocks, x, k))
            ensure(b == (16 // k) * 65536, f"k = {k}: {b:,} bayt; {(16 // k) * 65536:,} bekleniyordu")
            g = torch.autograd.grad(o.sum(), [x] + list(blocks.parameters()))
            for a, c in zip(g, ref_grad):
                close(a, c, atol=1e-6, what=f"k = {k} gradyan")
            rows.append([str(k), f"{b:,}"])
        R.table("İleri geçişte saklanan bayt (16 blok, B = 64, d = 256)", ["checkpoint aralığı", "bayt"], rows,
                note="Checkpoint'siz sayıma ağırlıklar da girer (matmul onları saklar; kopya değil, referans). Checkpoint'li sayım yalnız segment girdileri; geri geçişte segment başına yeniden hesaplanır.")
    R.test("saved_bytes + run_checkpointed", sb)
