import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("sign_agreement", "saturation")])

    def t():
        rng = np.random.default_rng(0)
        Z = rng.normal(0, 1.5, (2000, 10))
        delta = rng.normal(size=(2000, 5))
        As = 1 / (1 + np.exp(-Z))
        At = np.tanh(Z)
        ss, st = need(ns, "sign_agreement")(As, delta), ns["sign_agreement"](At, delta)
        close(ss, 1.0, what="sigmoid: girişler hep pozitif → bir nörona giden tüm ağırlık gradyanları aynı işaretli")
        ensure(st < 0.01, "tanh: sıfır merkezli → karışık işaret")
        rows = []
        for s in (1, 2, 4):
            Zs = rng.normal(0, s, 20000)
            rows.append([s, f"{need(ns, 'saturation')(Zs, 'sigmoid'):.3f}", f"{ns['saturation'](Zs, 'tanh'):.3f}"])
        R_.table("Ön-aktivasyon z ~ N(0, s²) iken doymuş nöron oranı (türev < maksimumun %10'u)", ["s", "sigmoid", "tanh"], rows,
                 note=f"Sigmoid'in iki kusuru: (1) çıkışı sıfır merkezli değil (0, 1): sonraki katmanda bir nörona gelen tüm ağırlıkların gradyanı δ'nın işaretini taşır (örneklerin %{ss * 100:.0f}'ında hepsi aynı işaretli) — ağırlıklar ancak zikzak yaparak farklı yönlere gidebilir; (2) türevi en fazla 0.25: her katmanda gradyanı en az 4 kat küçültür, doyma daha erken başlar. tanh sıfır merkezlidir (%{st * 100:.1f}) ve türevi 1'e kadar çıkar, ama o da büyük |z|'de doyar.")
    R_.test("sigmoid vs tanh", t)
