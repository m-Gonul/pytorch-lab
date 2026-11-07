import ast
import torch
import torch.nn.functional as F
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    src = ns["__source__"]

    def no_sdpa():
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == "attention":
                names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                ensure("scaled_dot_product_attention" not in names, "`attention` içinde F.scaled_dot_product_attention kullanma; elle yaz")
                return
        raise AssertionError("`attention` fonksiyonu yok")
    R.test("attention elle yazılmış", no_sdpa)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 3, 9, 8, generator=g, dtype=torch.float64) for _ in range(3))
    R.test("attention = SDPA", lambda: close(need(ns, "attention")(q, k, v), F.scaled_dot_product_attention(q, k, v)))
    R.test("attention = SDPA (is_causal)", lambda: close(need(ns, "attention")(q, k, v, causal=True), F.scaled_dot_product_attention(q, k, v, is_causal=True)))

    def vs_torch(causal):
        def f():
            torch.manual_seed(0)
            m = need(ns, "MultiHeadAttention", "sınıf")(16, 4).double()
            ensure(isinstance(getattr(m, "qkv", None), nn.Linear) and isinstance(getattr(m, "proj", None), nn.Linear), "self.qkv ve self.proj nn.Linear olmalı")
            ref = nn.MultiheadAttention(16, 4, batch_first=True).double()
            with torch.no_grad():
                ref.in_proj_weight.copy_(m.qkv.weight); ref.in_proj_bias.copy_(m.qkv.bias)
                ref.out_proj.weight.copy_(m.proj.weight); ref.out_proj.bias.copy_(m.proj.bias)
            x = torch.randn(3, 7, 16, generator=g, dtype=torch.float64)
            mask = torch.ones(7, 7, dtype=torch.bool).triu(1) if causal else None
            want, _ = ref(x, x, x, attn_mask=mask, need_weights=False)
            close(m(x, causal=causal), want)
        return f
    R.test("nn.MultiheadAttention ile aynı", vs_torch(False))
    R.test("nn.MultiheadAttention ile aynı (causal)", vs_torch(True))

    def grad():
        torch.manual_seed(1)
        m = need(ns, "MultiHeadAttention", "sınıf")(8, 2)
        m(torch.randn(2, 5, 8), causal=True).sum().backward()
        ensure(all(p.grad is not None for p in m.parameters()), "tüm parametreler gradyan almalı")
    R.test("geri geçiş", grad)
