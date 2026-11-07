import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    for f in ("quant_per_tensor", "quant_per_channel", "int8_matmul", "rel_error"):
        R.test(f"{f} döngüsüz", (lambda f=f: max_loops(src, 0, func=f)))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    W = rng.normal(size=(64, 128)); W[3] *= 100
    x = rng.normal(size=(16, 128))

    def types():
        q, s = need(ns, "quant_per_channel")(W)
        ensure(q.dtype == np.int8 and np.shape(s) == (64,), "int8 ve (out,) ölçek")
        ensure(np.abs(q).max(1).min() == 127, "her kanalın en büyük değeri 127'ye oturmalı")
        q2, s2 = need(ns, "quant_per_tensor")(W)
        ensure(q2.dtype == np.int8 and np.ndim(s2) == 0, "int8 ve skaler ölçek")
    R.test("dtype ve ölçek şekilleri", types)

    def mm():
        xq, sx = ns["quant_per_tensor"](x)
        wq, sw = ns["quant_per_channel"](W)
        want = (xq.astype(np.int64) @ wq.T.astype(np.int64)) * sx * sw
        close(need(ns, "int8_matmul")(xq, sx, wq, sw), want)
        big = np.full((1, 4096), 127, np.int8)
        r = ns["int8_matmul"](big, 1.0, big, 1.0)
        ensure(r[0, 0] == 127 * 127 * 4096, "int32 birikimi taşmamalı (int8/int16'da taşardı)")
    R.test("int8_matmul (int32 birikim)", mm)

    def cmp():
        xq, sx = ns["quant_per_tensor"](x)
        ref = x @ W.T
        e_t = need(ns, "rel_error")(ns["int8_matmul"](xq, sx, *ns["quant_per_tensor"](W)), ref)
        e_c = ns["rel_error"](ns["int8_matmul"](xq, sx, *ns["quant_per_channel"](W)), ref)
        mask = np.ones(64, bool); mask[3] = False
        e_t2 = ns["rel_error"](ns["int8_matmul"](xq, sx, *ns["quant_per_tensor"](W))[:, mask], ref[:, mask])
        e_c2 = ns["rel_error"](ns["int8_matmul"](xq, sx, *ns["quant_per_channel"](W))[:, mask], ref[:, mask])
        ensure(e_c2 < 0.2 * e_t2, "aykırı olmayan kanallarda kanal başına ölçek çok daha iyi olmalı")
        R.table("Aykırı kanallı W (64 × 128), göreli hata", ["ölçek", "tüm çıktı", "aykırı kanal hariç"],
                [["tensör başına", f"{e_t:.2e}", f"{e_t2:.2e}"], ["kanal başına", f"{e_c:.2e}", f"{e_c2:.2e}"]],
                note="Tüm çıktı hatası aykırı kanal tarafından domine ediliyor; asıl kayıp diğer kanallarda.")
    R.test("tensör başına vs kanal başına", cmp)
