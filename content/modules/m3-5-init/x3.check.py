import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "init_std")("glorot", 300, 100), np.sqrt(2 / 400), what="Glorot")
        close(ns["init_std"]("he", 300, 100), np.sqrt(2 / 300), what="He")
        close(need(ns, "uniform_limit")("glorot", 784, 300), np.sqrt(6 / (784 + 300)), what="Glorot uniform: √(6/(fan_in + fan_out))")
        sizes = [400, 100, 400, 50, 400]
        rows = []
        res = {}
        for kind in ("lecun", "glorot"):
            f, b = need(ns, "variance_flow")(sizes, kind)
            res[kind] = (f, b)
            rows.append([kind, " ".join(f"{v:.2f}" for v in f), " ".join(f"{v:.2f}" for v in b)])
        f, b = res["lecun"]
        ensure(all(abs(v - 1) < 0.1 for v in f) and max(b) > 3, "fan_in ileri sinyali korur, geri gradyanı korumaz")
        fg, bg = res["glorot"]
        ensure(max(abs(np.log(np.array(fg + bg)))) < max(abs(np.log(np.array(f + b)))), "fan_avg uzlaşması")
        R_.table("Doğrusal ağ 400-100-400-50-400: katman başına varyans oranı (1 = korunuyor)", ["başlatma", "ileri", "geri"], rows,
                 note="Var(çıkış) = fan_in · σ² · Var(giriş): σ² = 1/fan_in ileri sinyali korur. Geri yönde gradyan fan_out ile çarpılır: σ² = 1/fan_out gerekir. Katman genişlikleri farklıyken ikisi aynı anda sağlanamaz; Glorot & Bengio'nun uzlaşması fan_avg'dir — her iki yönde de oranlar 1'e en fazla birkaç kat uzak kalır. ReLU yarıyı sıfırladığı için He 2 çarpanını ekler.")
    R_.test("başlatma varyansları", t)
