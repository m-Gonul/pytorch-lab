import numpy as np
from skimage import data
from skimage.metrics import structural_similarity as sk
from labcheck import close, need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)
    x = data.camera().astype(float)
    y = np.clip(x + rng.normal(0, 15, x.shape), 0, 255)
    rows = []
    for win, sc in [(7, True), (7, False), (11, True), (3, False)]:
        def t(win=win, sc=sc):
            m, S = need(ns, "ssim_map")(x, y, win, sc)
            wm, wS = sk(x, y, win_size=win, use_sample_covariance=sc, data_range=255, full=True)
            close(m, wm, atol=1e-10, what="ortalama"); close(S, wS, atol=1e-10, what="harita")
            rows.append([win, "örnek" if sc else "popülasyon", f"{m:.4f}"])
        R.test(f"ssim_map = skimage (pencere {win}, {'örnek' if sc else 'popülasyon'} kovaryansı)", t)
    if rows:
        R.table("Aynı görüntü çifti, farklı SSIM tanımları", ["pencere", "kovaryans", "SSIM"], rows, note="Makaleleri karşılaştırırken hangi SSIM tanımının kullanıldığına bakın: sayılar değişir.")
