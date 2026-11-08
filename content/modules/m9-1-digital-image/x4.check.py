import numpy as np
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü yok", lambda: no_loops(src))
    rgb = data.astronaut()[60:316, 140:396].astype(np.float64) / 255

    def t():
        m = need(ns, "bayer_mask")(4, 4)
        ensure(m.shape == (4, 4, 3) and m[0, 0, 0] and m[0, 1, 1] and m[1, 0, 1] and m[1, 1, 2], "RGGB yerleşimi")
        ensure((m.sum(-1) == 1).all() and m[..., 1].sum() == 8, "her piksel tek renk, yeşil yarısı")
        raw = need(ns, "mosaic")(rgb)
        ensure(raw.shape == rgb.shape[:2], "ham görüntü tek kanal")
        close(raw[0, 0], rgb[0, 0, 0], what="(0,0) kırmızı örnek")
        close(raw[1, 1], rgb[1, 1, 2], what="(1,1) mavi örnek")
        out = need(ns, "demosaic_bilinear")(raw)
        mk = ns["bayer_mask"](*raw.shape)
        close(out[mk], rgb[mk], atol=1e-9, what="ölçülen renkler aynen korunur")
        P = need(ns, "psnr")
        inner = (slice(4, -4), slice(4, -4))
        per = [P(out[inner][..., c], rgb[inner][..., c]) for c in range(3)]
        near = np.repeat(np.repeat(raw[::2, ::2], 2, 0), 2, 1)
        nn = np.stack([np.repeat(np.repeat(raw[::2, ::2], 2, 0), 2, 1), np.repeat(np.repeat(raw[::2, 1::2], 2, 0), 2, 1), np.repeat(np.repeat(raw[1::2, 1::2], 2, 0), 2, 1)], -1)
        pn = [P(nn[inner][..., c], rgb[inner][..., c]) for c in range(3)]
        ensure(per[1] > per[0] and per[1] > per[2], f"yeşil kanal daha iyi kestirilmeli (2× örnek) {per}")
        ensure(min(per) > max(pn) - 1, "çift doğrusal, en yakın kopyadan iyi")
        lum = rgb.mean(-1)
        g = np.hypot(*np.gradient(lum))
        edge = g > np.quantile(g, 0.9)
        chroma_err = lambda im: np.abs((im - im.mean(-1, keepdims=True)) - (rgb - rgb.mean(-1, keepdims=True))).sum(-1)
        ce = chroma_err(out)
        R_.table("Demozaikleme (astronot kırpıntısı, 256 × 256)", ["yöntem", "PSNR R", "PSNR G", "PSNR B", "kenarlarda renk hatası", "düz bölgelerde renk hatası"],
                 [["en yakın (2 × 2 kopya)", *[f"{v:.1f} dB" for v in pn], f"{chroma_err(nn)[edge].mean():.3f}", f"{chroma_err(nn)[~edge].mean():.3f}"],
                  ["çift doğrusal", *[f"{v:.1f} dB" for v in per], f"{ce[edge].mean():.3f}", f"{ce[~edge].mean():.3f}"]],
                 note="Yeşil iki kat örneklendiği için en iyi kestirilen kanal. Renk (kroma) hatası kenarlarda yoğunlaşır: kırmızı ve mavi farklı konumlardan kestirildiği için keskin bir kenarın iki yanı farklı renk kanallarında farklı yerde 'görünür' — yanlış renk saçakları. Daha iyi yöntemler (Malvar–He–Cutler, kenar yönlü enterpolasyon) yeşilin yüksek frekansını kırmızı/maviye aktararak bunu azaltır.")
    R_.test("Bayer mozaiği ve çift doğrusal demozaikleme", t)
