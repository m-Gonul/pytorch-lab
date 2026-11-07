import time
import torch
from labcheck import close, need, ensure, fmt_time


def check(ns, R_):
    torch.manual_seed(0)
    V, d = 40, 32
    emb = torch.nn.Embedding(V, d)
    cell = torch.nn.GRUCell(d, d)
    head = torch.nn.Linear(d, V)
    with torch.no_grad():
        head.weight.mul_(3)

    @torch.no_grad()
    def step(tok, h):
        h2 = cell(emb(tok), h)
        return torch.log_softmax(head(h2), -1), h2

    def t():
        h0 = torch.randn(6, d)
        a_t, a_s = need(ns, "beam_loop")(step, h0, 0, 1, 3, 8)
        b_t, b_s = need(ns, "beam_batched")(step, h0, 0, 1, 3, 8)
        ensure(tuple(b_t.shape) == (6, 8), "(B, T) token")
        ensure(torch.equal(a_t, b_t), "döngü = toplu")
        close(a_s.numpy(), b_s.numpy(), atol=1e-4, what="skorlar")
        g_t, g_s = ns["beam_batched"](step, h0, 0, 1, 1, 8)
        ensure(bool((b_s >= g_s - 1e-5).all()), "beam skoru ≥ açgözlü skoru")
        rows = []
        for B, k in ((8, 3), (8, 8), (32, 5)):
            h0 = torch.randn(B, d)
            r = [B, k]
            for f in (ns["beam_loop"], ns["beam_batched"]):
                t0 = time.perf_counter()
                f(step, h0, 0, 1, k, 15)
                r.append(fmt_time(time.perf_counter() - t0))
            rows.append(r)
        gain = (b_s - g_s).mean().item()
        R_.table("Beam search, T = 15, V = 40 (CPU)", ["batch", "k", "döngü", "toplu"], rows,
                 note=f"Toplu sürüm B·k hipotezi tek bir step çağrısıyla ilerletir; aday seçimi (B, k·V) üzerinde tek topk, geri izleme gather ile. Bu örnekte k = 3 beam, açgözlüye göre ortalama {gain:.2f} nat daha yüksek log olasılık buldu.")
    R_.test("toplu beam search", t)
