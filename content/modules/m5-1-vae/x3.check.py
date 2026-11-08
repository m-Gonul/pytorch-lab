import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("lerp", "slerp", "typicality")])

    def t():
        rng = np.random.default_rng(0)
        z0, z1 = rng.normal(size=512), rng.normal(size=512)
        close(need(ns, "slerp")(z0, z1, [0, 1]), np.stack([z0, z1]), atol=1e-9, what="uç noktalar")
        close(need(ns, "lerp")(z0, z1, [0.5])[0], (z0 + z1) / 2, what="lerp")
        ts = np.linspace(0, 1, 5)
        L, S = ns["lerp"](z0, z1, ts), ns["slerp"](z0, z1, ts)
        tl, tsl = need(ns, "typicality")(L), ns["typicality"](S)
        ensure(tl[2] < 0.8 and abs(tsl[2] - 1) < 0.05, "lerp orta noktası atipik, slerp değil")
        rows = []
        for d in (2, 16, 128, 1024):
            a, b = rng.normal(size=d), rng.normal(size=d)
            rows.append([d, f"{float(ns['typicality'](ns['lerp'](a, b, [0.5]))[0]):.3f}", f"{float(ns['typicality'](ns['slerp'](a, b, [0.5]))[0]):.3f}"])
        R_.table("İki standart normal kodun orta noktası: norm/√d (önselden çekilmiş tipik kod ≈ 1)", ["d", "lerp", "slerp"], rows,
                 note="Bağımsız iki Gauss vektörü yüksek boyutta neredeyse diktir; ortalamalarının normu ≈ √(d/2): kodların ~%71'i büyüklüğünde, önselde neredeyse hiç görülmeyen bir bölgede. VAE'nin KL terimi kodları N(0, I)'ye çektiği için kod çözücü kabuk üzerinde eğitilmiştir — slerp, interpolasyon boyunca 'gerçekçi' kodlar verir (White 2016; GAN gizli uzaylarında da standart).")
    R_.test("interpolasyon", t)
