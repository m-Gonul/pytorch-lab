import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız adım döngüsü", lambda: max_loops(src, 1, func="train_fp16"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 10)).astype(np.float32)
    w_true = (1 + 0.003 * np.arange(10)).astype(np.float32)
    y = X @ w_true

    def stall():
        naive = need(ns, "train_fp16")(X, y, 0.01, 300, master=False)
        mast = ns["train_fp16"](X, y, 0.01, 300, master=True)
        start = float(np.mean((X @ np.ones(10, np.float32) - y) ** 2))
        ensure(naive > 0.3 * start, f"fp16 ağırlıklar takılmalıydı: başlangıç {start:.2e}, son {naive:.2e}")
        ensure(mast < 0.05 * naive, f"master kopya kaybı düşürmeli: {mast:.2e} vs {naive:.2e}")
        R.table("fp16 ağırlıklarla 300 GD adımı", ["", "kayıp"],
                [["başlangıç", f"{start:.2e}"], ["fp16 ağırlık", f"{naive:.2e}"], ["fp32 master + fp16 hesap", f"{mast:.2e}"]],
                note="Güncellemeler (~1e-5) ağırlığın yarım ulp'sinden (~4.9e-4) küçük: fp16'da kayboluyor.")
    R.test("fp16 ağırlıklar takılıyor, master kopya çözüyor", stall)

    def sg():
        a, b = need(ns, "scaled_grad")(1e-8, 65536.0)
        ensure(float(a) == 0.0, "ölçeksiz 1e-8 fp16'da 0 olmalı")
        ensure(abs(float(b) - 1e-8) / 1e-8 < 1e-3, f"ölçekli geri dönüş {float(b):.3e}")
    R.test("loss scaling alt taşmayı önlüyor", sg)
