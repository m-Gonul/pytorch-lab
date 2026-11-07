import torch
from labcheck import close, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]
    g = torch.Generator().manual_seed(0)

    def rep():
        x = torch.randn(10000, generator=g) * 100
        r = need(ns, "sr_bf16")(x, g)
        ensure(r.dtype == torch.float32, "fp32 tensör döndürmeli")
        ensure(torch.equal(r, r.to(torch.bfloat16).float()), "sonuç bf16'da tam temsil edilebilir olmalı")
        lo = x.to(torch.bfloat16).float()
        ensure(((r - x).abs() < 2 * (x.abs() * 2 ** -7 + 1e-30)).all(), "her değer komşu iki bf16 değerinden birine gitmeli")
    R.test("sonuçlar bf16'da temsil edilebilir", rep)

    def unbiased():
        for v in (1.001, -3.3333, 1e-3 + 1e-6):
            x = torch.full((200000,), v)
            m = need(ns, "sr_bf16")(x, g).double().mean().item()
            near = x.to(torch.bfloat16).double()[0].item()
            ensure(abs(m - v) < 0.1 * abs(near - v) + 1e-6 * abs(v), f"{v}: SR ortalaması {m}, en yakın {near}")
    R.test("SR yansız: E[SR(x)] ≈ x", unbiased)

    def train():
        X = torch.randn(512, 10, generator=g)
        wt = 1 + 0.003 * torch.arange(10.0)
        y = X @ wt
        start = ((X @ torch.ones(10) - y) ** 2).mean().item()
        nearest = need(ns, "train_bf16")(X, y, 0.01, 400, "nearest")
        stoch = ns["train_bf16"](X, y, 0.01, 400, "stochastic", torch.Generator().manual_seed(1))
        ensure(stoch < 0.2 * nearest, f"SR {stoch:.2e}, en yakın {nearest:.2e}")
        R.table("Saf bf16 ağırlıkla 400 GD adımı", ["yuvarlama", "son kayıp"],
                [["başlangıç", f"{start:.2e}"], ["en yakına", f"{nearest:.2e}"], ["stokastik", f"{stoch:.2e}"]],
                note="En yakına yuvarlamada küçük güncellemeler sürekli aşağı yuvarlanıp kayboluyor; SR'de beklenen değerde korunuyor.")
    R.test("SR'li bf16 eğitim takılmıyor", train)
