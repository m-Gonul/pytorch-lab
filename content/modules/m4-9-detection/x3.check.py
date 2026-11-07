import numpy as np
from labcheck import close, need, ensure


def _hard(b, s, thr, iou_m):
    order = np.argsort(-s, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][iou_m(b[i][None], b[order[1:]])[0] <= thr]
    return np.array(keep), s[keep]


def check(ns, R_):
    iou_m = ns["iou_m"]

    def t_basic():
        b = np.array([[0., 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]])
        k, s = need(ns, "soft_nms")(b, np.array([.9, .8, .7]), 0.5)
        o = 9 * 10 / (200 - 90)
        ensure(list(k) == [0, 2, 1], f"sıra {list(k)}")
        close(s, np.array([.9, .7, .8 * np.exp(-o ** 2 / .5)]), rtol=1e-9, what="Gauss azaltma")
        gts = [np.array([[0., 0, 10, 10], [20, 0, 30, 10]])]
        d = [(np.array([[0., 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10]]), np.array([.9, .8, .7]))]
        close(need(ns, "match_ap")(d, gts, 0.5), 0.5 * 1 + 0.5 * 2 / 3, what="AP")
    R_.test("Soft-NMS ve AP", t_basic)

    def t_crowd():
        rng = np.random.default_rng(0)
        gts, raw = [], []
        for _ in range(200):
            c = rng.uniform(20, 80, 2)
            w = rng.uniform(20, 30)
            g1 = np.array([c[0] - w / 2, c[1] - w / 2, c[0] + w / 2, c[1] + w / 2])
            sh = w * rng.uniform(0.15, 0.3)
            g = np.stack([g1, g1 + np.array([sh, sh * 0.5, sh, sh * 0.5])])
            gts.append(g)
            b = np.concatenate([g + rng.normal(0, 1.0, g.shape), g[[0, 0, 1]] + rng.normal(0, 2.5, (3, 4))])
            s = np.array([rng.uniform(0.85, 1), rng.uniform(0.5, 0.65), rng.uniform(0.7, 0.85), rng.uniform(0.7, 0.85), rng.uniform(0.3, 0.45)])
            raw.append((b, s))
        thrs = 0.5 + 0.05 * np.arange(10)
        res, rows = {}, []
        for name, f in (("NMS yok", lambda b, s: (np.arange(len(s)), s)), ("NMS 0.5", lambda b, s: _hard(b, s, .5, iou_m)),
                        ("NMS 0.7", lambda b, s: _hard(b, s, .7, iou_m)), ("Soft-NMS σ = 0.5", ns["soft_nms"])):
            d = [(b[k], s2) for (b, s) in raw for k, s2 in [f(b, s)]]
            a = [need(ns, "match_ap")(d, gts, t) for t in thrs]
            res[name] = a[0]
            rows.append([name, f"{a[0]:.3f}", f"{a[5]:.3f}", f"{np.mean(a):.3f}"])
        ensure(res["Soft-NMS σ = 0.5"] > res["NMS 0.5"] + 0.1 and res["Soft-NMS σ = 0.5"] > res["NMS yok"], "Soft-NMS kalabalıkta kazanmalı")
        R_.table("Kalabalık çiftler (gerçek kutular arası IoU ≈ 0.52), 200 sahne", ["son işlem", "AP@0.5", "AP@0.75", "AP@[.50:.95]"], rows,
                 note="Arkadaki nesnenin tespiti öndekiyle IoU ≈ 0.5 olduğu için NMS 0.5 onu siler (recall kaybı); eşik 0.7 kopyaları kaçırır. Soft-NMS silmez, skoru IoU ile sürekli düşürür: kopyalar (IoU yüksek) dibe iner, komşu nesne (IoU orta) sıralamada kalır.")
    R_.test("kalabalık sahne", t_crowd)
