import math
import torch
import torch.nn.functional as F
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("blocked_attention'da tek döngü (bloklar)", lambda: max_loops(src, 1, func="blocked_attention", why="satır ya da eleman döngüsü yok"))

    def ast_no_sdpa():
        import ast
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == "blocked_attention":
                names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
                ensure("scaled_dot_product_attention" not in names and "softmax" not in names,
                       "SDPA ya da softmax çağırma; online softmax'ı elle yaz")
                return
        raise AssertionError("`blocked_attention` yok")
    R.test("SDPA/softmax çağrısı yok", ast_no_sdpa)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 3, 150, 16, generator=g, dtype=torch.float64) for _ in range(3))

    def eq(block, causal):
        return lambda: close(need(ns, "blocked_attention")(q, k, v, block=block, causal=causal),
                             F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    R.test("SDPA ile aynı (blok 64, 150 bölünmüyor)", eq(64, False))
    R.test("SDPA ile aynı (causal, blok 32)", eq(32, True))
    R.test("blok 1 (en uç durum, causal)", eq(1, True))

    def big():
        qq = q * 30
        close(need(ns, "blocked_attention")(qq, k, v, block=40), F.scaled_dot_product_attention(qq, k, v), what="büyük skorlar")
    R.test("büyük skorlarda taşma yok", big)

    def grad():
        a = [t.clone().requires_grad_(True) for t in (q, k, v)]
        b = [t.clone().requires_grad_(True) for t in (q, k, v)]
        need(ns, "blocked_attention")(*a, block=48, causal=True).pow(2).sum().backward()
        F.scaled_dot_product_attention(*b, is_causal=True).pow(2).sum().backward()
        for x, y, n in zip(a, b, "qkv"):
            close(x.grad, y.grad, what=f"d{n}")
    R.test("gradyanlar SDPA ile aynı", grad)
    if "blocked_attention" not in ns:
        return
    T = 2048
    Q, K, V = (torch.randn(1, 8, T, 64, generator=g) for _ in range(3))

    def naive():
        return ((Q @ K.transpose(-2, -1)) / 8).softmax(-1) @ V
    t_n = bench(naive, repeat=3)
    t_b = bench(lambda: ns["blocked_attention"](Q, K, V, block=256), repeat=3)
    t_s = bench(lambda: F.scaled_dot_product_attention(Q, K, V), repeat=3)
    mb = lambda n: f"{n * 4 / 2**20:.0f} MB"
    R.table(f"T = {T}, 8 head, d_k = 64", ["yöntem", "süre", "skor ara belleği"],
            [["naif (tam T×T)", fmt_time(t_n), mb(8 * T * T)],
             ["senin blocked (blok 256)", fmt_time(t_b), mb(8 * T * 256)],
             ["F.scaled_dot_product_attention", fmt_time(t_s), "fused"]],
            note="Bloklu yol CPU'da bile hızlanabilir: blok önbelleğe sığar. GPU'da aynı fikir blokları SRAM'de tutup HBM trafiğini keser (FlashAttention).")
