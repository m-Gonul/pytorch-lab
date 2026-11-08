import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        eigs = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25])
        ensure(need(ns, "active_dims")(eigs, 1.0, 1.0) == 3 and ns["active_dims"](eigs, 1.0, 0.1) == 6, "aktif boyutlar")
        close(need(ns, "decoder_norms")(eigs, 1.0, 1.0), np.sqrt(np.maximum(0, eigs - 1)), what="kod çözücü normları")
        rng = np.random.default_rng(0)
        d, n, s2 = 6, 200000, 1.0
        Wt = np.diag(np.sqrt(np.maximum(eigs - s2, 0)))
        Z = rng.normal(size=(n, d))
        X = Z @ Wt.T + rng.normal(0, np.sqrt(s2), (n, d))
        lam = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
        rows = []
        for beta in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0):
            kl = need(ns, "kl_per_dim")(lam, s2, beta)
            rows.append([beta, ns["active_dims"](lam, s2, beta), f"{kl.sum():.3f}", " ".join(f"{v:.2f}" for v in ns["decoder_norms"](lam, s2, beta))])
        ensure(rows[0][1] >= 5 and rows[-1][1] == 0, "β büyüdükçe tüm boyutlar çöker")
        R_.table("Doğrusal VAE optimumu, gözlem gürültüsü σ² = 1, veri özdeğerleri ≈ 8, 4, 2, 1, 1, 1", ["β", "aktif boyut", "toplam KL (nat)", "kod çözücü sütun normları"], rows,
                 note="KL terimi bir 'bilgi bütçesi'dir: bir gizli boyutu kullanmanın bedeli, yeniden yapılandırmada sağladığı kazancı (λ_i − βσ²) aşarsa o boyut çöker — posterior önsele eşitlenir. β-VAE (β > 1) bu yüzden daha az ve daha 'ayrık' boyut kullanır; güçlü (otoregresif) kod çözücülerde tüm boyutların çökmesi (posterior çöküşü) yaygındır, KL ısınması (β'yı 0'dan 1'e artırmak) ya da free bits bunu hafifletir.")
    R_.test("β-VAE ve çöküş", t)
