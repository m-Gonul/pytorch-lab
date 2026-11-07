import random
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t_iou():
        iou = need(ns, "iou")
        close(iou((1, 1, 5, 5), (3, 2, 7, 6)), 6 / 26, what="IoU")
        close(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0, what="ayrık kutular")
        close(iou((0, 0, 4, 4), (1, 1, 3, 3)), 0.25, what="iç içe")
        close(iou((0, 0, 2, 2), (0, 0, 2, 2)), 1.0, what="özdeş")
        B = [(0, 0, 10, 10), (1, 1, 11, 11), (20, 20, 30, 30), (8, 8, 18, 18), (21, 19, 31, 29)]
        ensure(need(ns, "nms")(B, [0.9, 0.8, 0.7, 0.6, 0.5], 0.5) == [0, 2, 3], "NMS örneği")
        ensure(ns["nms"](B, [0.5, 0.8, 0.7, 0.6, 0.9], 0.5) == [4, 1, 3], "skora göre sıralı")
    R_.test("IoU ve NMS", t_iou)

    def t_ap():
        prec, rec = need(ns, "pr_curve")([1, 1, 0, 1, 0, 0, 1, 0], 5)
        close(prec[3], 0.75, what="precision@4")
        close(need(ns, "ap11")(prec, rec), (6.5 + 8 / 7) / 11, what="11 noktalı AP")
        close(need(ns, "ap_all")(prec, rec), 0.2 * (2.75 + 4 / 7), what="tüm noktalı AP")
        gts = [(0, 0, 10, 10), (20, 0, 30, 10)]
        dets = [(0, 0, 10, 10), (1, 0, 11, 10), (20, 0, 30, 10), (50, 50, 60, 60)]
        ensure(need(ns, "match")(dets, [0.9, 0.8, 0.7, 0.6], gts, 0.5) == [True, False, True, False], "yinelenen tespit yanlış")
        ensure(ns["match"](dets, [0.6, 0.9, 0.7, 0.8], gts, 0.5) == [True, False, True, False], "skor sırası: (1,0,11,10) önce eşleşir")
        rng = random.Random(0)
        rows = []
        for name, tp in (("mükemmel sıralama", [1] * 6 + [0] * 6), ("rastgele sıralama", [rng.random() < 0.5 for _ in range(12)]), ("ters sıralama", [0] * 6 + [1] * 6)):
            p, r = ns["pr_curve"](tp, sum(tp))
            rows.append([name, f"{ns['ap11'](p, r):.3f}", f"{ns['ap_all'](p, r):.3f}"])
        R_.table("Aynı 12 tespit, farklı skor sıralaması", ["sıralama", "AP (11 nokta)", "AP (tüm noktalar)"], rows,
                 note="AP yalnız 'doğru tespitler yanlışlardan önce mi geliyor' sorusuna bakar: skorların mutlak değeri değil, sıralaması önemli. Doğru tespit sayısı aynı olsa da sıralama kötüleştikçe AP düşer; 11 noktalı biçim kaba bir yaklaşıklıktır.")
    R_.test("eşleştirme ve AP", t_ap)
