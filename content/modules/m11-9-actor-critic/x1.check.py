from labcheck import need, ensure


def check(ns, R_):
    def t():
        mus, sig = need(ns, "gauss_reinforce")(400, 64, 0.05, 1)
        ensure(abs(mus[-1] - 3) < 0.15 and sig[-1] < 0.5 * sig[0], f"μ → 3 ve σ küçülmeli ({mus[-1]:.2f}, {sig[-1]:.2f})")
        R_.table("Gauss politikası, ödül −(a − 3)²", ["yineleme", "μ", "σ"], [[k, f"{mus[k]:.3f}", f"{sig[k]:.3f}"] for k in (0, 25, 50, 100, 200, 400)],
                 note="σ keşif miktarıdır ve kendiliğinden azalır: optimuma yaklaştıkça geniş σ ortalama ödülü düşürür. Doğal gradyan (σ² ile ölçekleme) adımları ölçek-bağımsız kılar.")
    R_.test("Gauss REINFORCE", t)
