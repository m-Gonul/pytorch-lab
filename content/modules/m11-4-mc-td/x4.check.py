import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t_sel():
        rng = np.random.default_rng(0)
        sel = need(ns, "select")
        q = np.array([0.0, 1.0, 1.0])
        picks = [sel(q, "greedy", 0, rng) for _ in range(400)]
        ensure(set(picks) == {1, 2}, "greedy: eşitlikleri rastgele boz (hep ilkini seçme)")
        picks = np.array([sel(np.array([0.0, 1.0]), "eps", 0, rng, eps=0.5) for _ in range(4000)])
        ensure(abs((picks == 0).mean() - 0.25) < 0.03, "ε = 0.5: kötü eylem ≈ ε/2 olasılıkla")
        picks = np.array([sel(np.array([0.0, 0.1]), "boltzmann", 0, rng, tau=0.1) for _ in range(4000)])
        ensure(abs((picks == 1).mean() - 1 / (1 + np.exp(-1))) < 0.03, "Boltzmann: p ∝ exp(q/τ)")
        early = np.array([sel(np.array([0.0, 1.0]), "eps-decay", 0, rng) for _ in range(2000)])
        late = np.array([sel(np.array([0.0, 1.0]), "eps-decay", 400, rng) for _ in range(2000)])
        ensure((early == 0).mean() > 0.4 and (late == 0).mean() < 0.06, "azalan ε: başta ≈ 1, sonra 0.05")
    R_.test("eylem seçimi", t_sel)

    def t_chain():
        ql = need(ns, "q_learning")
        rows, res = [], {}
        for name, m, q0 in (("ε-greedy, ε = 0.1", "eps", 0.0), ("azalan ε (1 → 0.05)", "eps-decay", 0.0), ("Boltzmann, τ = 0.1", "boltzmann", 0.0), ("iyimser başlangıç Q₀ = 20, greedy", "greedy", 20.0)):
            r = [name]
            for N in (6, 10):
                out = [ql(N, 250, m, q0, s) for s in range(8)]
                found = np.mean([f is not None for _, f in out])
                res[(m, N)] = found
                r += [f"{found:.2f}", f"{np.mean([g[-50:].mean() for g, _ in out]):.2f}"]
            rows.append(r)
        ensure(res[("greedy", 10)] == 1.0 and res[("eps", 10)] < 0.5, f"uzun zincirde iyimser başlangıç bulur, sabit ε çoğunlukla bulamaz {res}")
        R_.table("Zincir ortamında keşif (8 tohum, 250 bölüm): uzak ödülü bulma oranı ve son 50 bölümün ortalama getirisi", ["yöntem", "N = 6: buldu", "N = 6: getiri", "N = 10: buldu", "N = 10: getiri"], rows,
                 note="Soldaki küçük ödül (0.05) hemen bulunur ve açgözlü politikayı oraya kilitler. Sağ uca ulaşmak için art arda N − 2 kez sağa gitmek gerekir: rastgele (yönsüz) keşifle olasılık ≈ 2^−(N−2), zincir uzadıkça üstel olarak küçülür. İyimser başlangıç denenmemiş her eylemi 'cazip' kılar: ajan sistematik olarak dener, değer gerçekleri gördükçe iner (belirsizlik karşısında iyimserlik — UCB'nin de ilkesi, M11.7). Sabit ε'lu ajan uzak ödülü bulsa bile ε oranında rastgele davrandığı için getirisi düşük kalır.")
    R_.test("keşif yöntemleri", t_chain)
