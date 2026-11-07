import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = data.camera()
    ii = cv2.integral(img).astype(np.float64)
    im = img.astype(np.int64)

    def bx():
        close(need(ns, "box_sum")(ii, 10, 20, 7, 9), im[10:17, 20:29].sum(), what="kutu toplamı")
        dx, dy = need(ns, "haar")(ii, 100, 200, 3)
        close(dx, im[97:103, 200:203].sum() - im[97:103, 197:200].sum(), what="dx")
        close(dy, im[100:103, 197:203].sum() - im[97:100, 197:203].sum(), what="dy")
    R.test("box_sum ve haar", bx)

    rng = np.random.default_rng(0)
    pts = np.stack([rng.integers(60, 450, 200), rng.integers(60, 450, 200)], 1)

    def inv():
        d1 = need(ns, "usurf")(img, pts, 2)
        ensure(d1.shape == (200, 64), "(N, 64)")
        img2 = np.clip(img.astype(float) * 0.6 + 40, 0, 255).astype(np.uint8)
        d2 = ns["usurf"](img2, pts, 2)
        cos = (d1 * d2).sum(1)
        acc = ((d1 @ d2.T).argmax(1) == np.arange(200)).mean()
        raw = lambda im: np.stack([im[y - 20:y + 20, x - 20:x + 20].astype(float).ravel() for y, x in pts])
        r1, r2 = raw(img), raw(img2)
        accr = (np.argmin(((r1[:, None] - r2[None]) ** 2).sum(-1), 1) == np.arange(200)).mean()
        ensure(np.median(cos) > 0.98 and acc > accr + 0.2, f"medyan kosinüs {np.median(cos):.3f}, doğruluk {acc:.2f} / ham {accr:.2f}")
        R.table("Parlaklık ve kontrast değişimi (0.6·I + 40, uint8'e yuvarlanmış)", ["betimleyici", "eşleşme doğruluğu"], [["U-SURF (64)", f"{acc:.2f}"], ["ham 40 × 40 yama, SSD", f"{accr:.2f}"]],
                note=f"Kosinüs medyanı {np.median(cos):.3f}, en küçüğü {cos.min():.3f}: düz bölgelerde Haar yanıtları küçük olduğu için uint8 yuvarlama gürültüsü baskınlaşır.")
    R.test("aydınlatma değişmezliği", inv)
