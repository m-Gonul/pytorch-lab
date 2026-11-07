import time
import torch
from labcheck import close, need, ensure


def check(ns, R_):
    anchors = torch.tensor([[11.0, 11.0], [21.0, 21.0]])
    X, T = need(ns, "shapes_dataset")(1500, 0)
    Xt, Tt = ns["shapes_dataset"](300, 1)

    def t_code():
        a = torch.tensor([[0., 0, 10, 10], [5, 5, 15, 15]])
        close(need(ns, "box_iou")(a, a).numpy(), torch.tensor([[1, 25 / 175], [25 / 175, 1]]).numpy(), atol=1e-6, what="box_iou")
        for k in range(20):
            b, l = T[k]
            t = need(ns, "encode")(b, l, 8, anchors, 3)
            ensure(int(t[..., 4].sum()) == len(b), "her nesne tam bir çapaya")
            p = torch.zeros_like(t)
            p[..., :2] = torch.logit(t[..., :2].clamp(1e-6, 1 - 1e-6))
            p[..., 2:4] = t[..., 2:4]
            p[..., 4] = torch.where(t[..., 4] > 0, 20.0, -20.0)
            p[..., 5:] = 20 * t[..., 5:]
            db, ds, dl = need(ns, "decode")(p, anchors, 64, 0.5)
            o = torch.argsort(db[:, 0] * 1000 + db[:, 1])
            g = torch.argsort(b[:, 0] * 1000 + b[:, 1])
            close(db[o].numpy(), b[g].numpy(), atol=1e-3, what="decode(encode) = kutular")
            ensure(torch.equal(dl[o], l[g]), "etiketler")
        b = torch.tensor([[0., 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [8, 8, 18, 18], [21, 19, 31, 29]])
        ensure(need(ns, "nms")(b, torch.tensor([.9, .8, .7, .6, .5]), 0.5).tolist() == [0, 2, 3], "NMS")
    R_.test("kodlama / çözme / NMS", t_code)

    def t_train():
        torch.manual_seed(0)
        m = need(ns, "TinyYOLO")(2, 3)
        ensure(tuple(m(X[:2]).shape) == (2, 8, 8, 2, 8), "çıkış (N, S, S, A, 5 + C)")
        Y = torch.stack([ns["encode"](b, l, 8, anchors, 3) for b, l in T])
        opt = torch.optim.Adam(m.parameters(), 3e-3)
        t0 = time.perf_counter()
        hist = []
        for ep in range(6):
            m.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                idx = perm[i:i + 64]
                loss = need(ns, "yolo_loss")(m(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            hist.append(float(loss))
        m.eval()
        with torch.no_grad():
            P = m(Xt)
        rows = []
        res = {}
        for name, use_nms in (("NMS yok", False), ("NMS 0.5", True)):
            dets = []
            for k in range(len(Xt)):
                b, s, l = ns["decode"](P[k], anchors, 64, 0.05)
                if use_nms and len(b):
                    keep = ns["nms"](b, s, 0.5)
                    b, s, l = b[keep], s[keep], l[keep]
                dets.append((b, s, l))
            a50 = need(ns, "mean_ap")(dets, Tt, 3, 0.5)
            a75 = ns["mean_ap"](dets, Tt, 3, 0.75)
            coco = sum(ns["mean_ap"](dets, Tt, 3, 0.5 + 0.05 * k) for k in range(10)) / 10
            res[name] = a50
            rows.append([name, f"{sum(len(d[0]) for d in dets) / len(dets):.1f}", f"{a50:.3f}", f"{a75:.3f}", f"{coco:.3f}"])
        ensure(res["NMS 0.5"] > 0.75, f"mAP@0.5 = {res['NMS 0.5']:.3f} (> 0.75 beklenir)")
        R_.table(f"TinyYOLO, 1500 görüntü × 6 epoch ({time.perf_counter() - t0:.0f} sn, kayıp {hist[0]:.2f} → {hist[-1]:.2f})", ["sonrası", "görüntü başı kutu", "mAP@0.5", "mAP@0.75", "AP@[.50:.95]"], rows,
                 note="Görüntü başına 1–3 nesne var. NMS'siz komşu hücre/çapa kopyaları yanlış pozitif olur. mAP@0.75 ve COCO ölçütü konum hassasiyetini cezalandırır: 8 × 8 ızgaradaki kaba dedektör 'nesneyi bulmakta' iyi, 'kutuyu tam oturtmakta' daha zayıf.")
    R_.test("eğitim ve mAP", t_train)
