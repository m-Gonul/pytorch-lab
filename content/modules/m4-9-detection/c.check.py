import numpy as np
from labcheck import close, max_loops, need, ensure


def _ref_nms(b, s, thr):
    def iou(a, c):
        iw = max(0, min(a[2], c[2]) - max(a[0], c[0])); ih = max(0, min(a[3], c[3]) - max(a[1], c[1]))
        i = iw * ih
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i)
    keep = []
    for i in np.argsort(-s, kind="stable"):
        if all(iou(b[i], b[j]) <= thr for j in keep):
            keep.append(int(i))
    return keep


def _boxes(rng, n):
    c = rng.uniform(0, 100, (n, 2)); wh = rng.uniform(5, 30, (n, 2))
    return np.concatenate([c - wh / 2, c + wh / 2], 1)


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüler", lambda: ([max_loops(src, 0, func=f) for f in ("iou_matrix", "fast_nms", "ap_all")], max_loops(src, 1, func="nms"), max_loops(src, 1, func="anchor_kmeans")))
    rng = np.random.default_rng(0)

    def t_nms():
        A, B = _boxes(rng, 7), _boxes(rng, 5)
        M = need(ns, "iou_matrix")(A, B)
        ensure(M.shape == (7, 5) and (M >= 0).all() and (M <= 1).all(), "IoU matrisi")
        close(np.diag(ns["iou_matrix"](A, A)), np.ones(7), what="köşegen 1")
        for seed in range(5):
            r = np.random.default_rng(seed)
            b, s = _boxes(r, 120), r.random(120)
            g = list(need(ns, "nms")(b, s, 0.4))
            ensure(g == _ref_nms(b, s, 0.4), "açgözlü NMS referansla aynı")
            f = set(need(ns, "fast_nms")(b, s, 0.4).tolist())
            ensure(f <= set(g), "Fast NMS ⊆ açgözlü NMS")
        b, s = _boxes(rng, 400), rng.random(400)
        R_.note(f"400 rastgele kutu, eşik 0.4: açgözlü NMS {len(ns['nms'](b, s, .4))}, Fast NMS {len(ns['fast_nms'](b, s, .4))} kutu tutuyor — Fast NMS bastırılmış kutuların da bastırmasına izin verdiği için biraz daha fazla siler; karşılığında tamamen paralel.")
    R_.test("NMS", t_nms)

    def t_ap():
        close(need(ns, "ap_all")(np.array([.9, .8, .7, .6, .5, .4, .3, .2]), np.array([1, 1, 0, 1, 0, 0, 1, 0.]), 5), 0.2 * (2.75 + 4 / 7), what="AP")
        close(ns["ap_all"](np.array([.2, .9, .4, .8, .5, .6, .3, .7]), np.array([0, 1, 0, 1, 0, 1, 1, 0.]), 5), 0.2 * (2.75 + 4 / 7), what="sırasız girdi")
    R_.test("AP", t_ap)

    def t_km():
        r = np.random.default_rng(1)
        ped = np.stack([r.normal(12, 1.5, 300), r.normal(32, 3, 300)], 1)
        car = np.stack([r.normal(40, 4, 300), r.normal(18, 2, 300)], 1)
        sq = np.stack([r.normal(20, 2, 300), r.normal(20, 2, 300)], 1)
        wh = np.abs(np.concatenate([ped, car, sq]))
        A, miou = need(ns, "anchor_kmeans")(wh, 3, 30, 0)
        A = A[np.argsort(A[:, 0])]
        close(A, np.array([[12, 32], [20, 20], [40, 18]]), atol=2.5, what="üç çapa: yaya, kare, araba")
        rows = []
        for k in (1, 2, 3, 5, 9):
            rows.append([k, f"{ns['anchor_kmeans'](wh, k, 30, 0)[1]:.3f}"])
        R_.table("Çapa sayısı k ve ortalama en iyi IoU", ["k", "ort. IoU"], rows,
                 note="YOLOv2 Tablo 1'in küçük hâli: çapa sayısı arttıkça kutuların en yakın çapaya IoU'su artar, ama her çapa ağın çıkışına 5 + C sayı ekler. Öklid uzaklığı büyük kutuları kayırırdı; 1 − IoU ölçekten bağımsızdır.")
    R_.test("çapa k-means", t_km)
