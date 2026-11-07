import numpy as np
import cv2
from labcheck import close, need, ensure


def check(ns, R_):
    rng = np.random.default_rng(1)

    def pp():
        a = np.zeros((210, 160, 3), np.uint8); b = np.zeros_like(a)
        a[100:110, 50:60] = (200, 50, 50)
        b[30:40, 80:90] = (10, 250, 10)
        out = need(ns, "preprocess")(a, b)
        ensure(out.shape == (84, 84) and out.dtype == np.uint8, "84×84 uint8")
        ensure(out.max() > 100, "maksimum: iki karedeki nesneler de görünmeli")
        x = rng.integers(0, 256, (2, 210, 160, 3), dtype=np.uint8)
        ref = cv2.resize(cv2.cvtColor(np.maximum(x[0], x[1]), cv2.COLOR_RGB2GRAY), (84, 110), interpolation=cv2.INTER_AREA)[18:102]
        d = np.abs(ns["preprocess"](x[0], x[1]).astype(int) - ref.astype(int)).max()
        ensure(d <= 1, f"OpenCV gri+küçültme ile en fazla 1 fark (bulunan {d})")
        R_.table("Ön işleme", ["aşama", "boyut", "bayt"], [["ham kare", "210×160×3", 210 * 160 * 3], ["gri + 110×84", "110×84", 110 * 84], ["kırpılmış", "84×84", 84 * 84], ["4 kare yığını", "4×84×84", 4 * 84 * 84]],
                 note="1 M geçişlik tampon float32 ile ~56 GB; uint8 kareler ve s/s′ paylaşımıyla ~7 GB. Maksimum alma, Atari'nin bazı nesneleri yalnız tek karelerde çizmesine karşıdır.")
    R_.test("preprocess", pp)

    def fs():
        S = need(ns, "FrameStack")(4)
        f = [np.full((84, 84), i, np.uint8) for i in range(6)]
        S.reset(f[0])
        ensure(S.state().shape == (4, 84, 84) and (S.state() == 0).all(), "reset k kopya")
        for i in range(1, 6):
            S.push(f[i])
        ensure([int(S.state()[j, 0, 0]) for j in range(4)] == [2, 3, 4, 5], "en eski atılır, sıra korunur")
        ensure([need(ns, "clip_reward")(r) for r in (-7, 0, 0.3, 400)] == [-1, 0, 1, 1], "ödül kırpma")
    R_.test("FrameStack ve clip_reward", fs)
