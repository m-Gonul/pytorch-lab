import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


@torch.no_grad()
def greedy_naive(model, idx, n):
    for _ in range(n):
        idx = torch.cat([idx, model(idx)[:, -1].argmax(-1, keepdim=True)], 1)
    return idx


def check(ns, R):
    src = ns["__source__"]
    R.test("forward_step: tek döngü", lambda: max_loops(src, 1, func="forward_step", why="yalnız bloklar"))
    R.test("generate_kv: tek döngü", lambda: max_loops(src, 1, func="generate_kv", why="yalnız adımlar"))
    TinyGPT = need(ns, "TinyGPT", "sınıf")
    torch.manual_seed(0)
    m = TinyGPT().eval()
    for p in m.parameters():
        p.data = p.data.double()
    idx = torch.randint(0, 64, (2, 15))

    def prefill():
        with torch.no_grad():
            logits, cache = need(ns, "forward_step")(m, idx, [None] * 4, 0)
        close(logits, m(idx), what="prefill logit'leri")
        ensure(len(cache) == 4 and tuple(cache[0][0].shape) == (2, 4, 15, 16), "cache: blok başına (K, V), (B, h, S, dk)")
    R.test("prefill = tam ileri geçiş", prefill)

    def steps():
        full = m(idx)
        with torch.no_grad():
            lg, cache = need(ns, "forward_step")(m, idx[:, :6], [None] * 4, 0)
            close(lg, full[:, :6], what="ilk 6")
            lg, cache = ns["forward_step"](m, idx[:, 6:9], cache, 6)
            close(lg, full[:, 6:9], what="3'lü parça (6–8)")
            for t in range(9, 15):
                lg, cache = ns["forward_step"](m, idx[:, t:t + 1], cache, t)
                close(lg[:, 0], full[:, t], what=f"konum {t}")
    R.test("artımlı adımlar her konumda tam geçişle aynı", steps)

    def gen():
        out = need(ns, "generate_kv")(m, idx[:1, :5], 30)
        ensure(torch.equal(out, greedy_naive(m, idx[:1, :5], 30)), "cache'siz greedy ile farklı dizi")
    R.test("generate_kv = cache'siz greedy", gen)
    if "generate_kv" not in ns:
        return
    torch.manual_seed(0)
    mf = TinyGPT(d=128, L=4, h=4).eval()
    p = torch.randint(0, 64, (1, 16))
    n = 200
    t_naive = bench(lambda: greedy_naive(mf, p, n), repeat=1, number=1)
    t_kv = bench(lambda: ns["generate_kv"](mf, p, n), repeat=1, number=1)
    R.table(f"{n} token greedy, d = 128, 4 blok", ["yöntem", "süre", "token/s"],
            [["her adımda tüm diziyi yeniden hesapla", fmt_time(t_naive), f"{n / t_naive:,.0f}"],
             ["senin KV-cache", fmt_time(t_kv), f"{n / t_kv:,.0f}"]],
            note="Naif yol adım t'de O(t) token işler → toplam O(n²). Cache bellek ister: 2 · blok · h · S · dk sayı.")
