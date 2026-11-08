import math
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        e = [0.0]
        for _ in range(1000):
            e = need(ns, "ema_update")(e, [1.0], 0.999)
        close(e[0], 1 - 0.999 ** 1000, what="EMA: sabit hedefe yaklaşma")
        emb = need(ns, "timestep_embedding")(50, 8)
        close(emb[1], math.sin(5), what="ω₁ = 0.1")
        close(emb[4:], [math.cos(50 * 10000 ** (-i / 4)) for i in range(4)], what="cos yarısı")
        ts = need(ns, "ddim_timesteps")(200, 25)
        ensure(ts[0] == 199 and ts[-1] == 0 and len(ts) == 25 and all(a > b for a, b in zip(ts, ts[1:])), "azalan alt dizi")
        ensure(need(ns, "nfe")(25, True, False) == 50 and ns["nfe"](25, True, True) == 25 and ns["nfe"](200, False, False) == 200, "NFE")
        close(need(ns, "dice")([1] * 30 + [0] * 20, [1] * 25 + [0] * 5 + [1] * 15 + [0] * 5), 2 * 25 / 70, what="Dice")
        ensure(ns["dice"]([0, 0], [0, 0]) == 1.0, "boş-boş")
        rows = []
        for d in (0.9, 0.99, 0.995, 0.999, 0.9999):
            rows.append([d, f"{1 / (1 - d):.0f}", f"{d ** 1000:.3f}"])
        R_.table("EMA bozunması", ["β", "etkin pencere (adım)", "1000 adım önceki ağırlığın payı"], rows,
                 note="Kısa eğitimde (≈ 1000 adım) 0.9999 neredeyse başlangıç ağırlıklarında kalır — EMA'yı 'ısıtmak' (β'yı adım sayısıyla artırmak, ör. min(β, (1 + n)/(10 + n))) ya da eğitim süresine uygun β seçmek gerekir.")
    R_.test("bütçe hesapları", t)
