import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        B = need(ns, "ReplayBuffer")(5)
        B.add(np.arange(3.0)[:, None])
        ensure(len(B) == 3, "uzunluk")
        B.add(np.arange(3.0, 7.0)[:, None])
        ensure(len(B) == 5 and sorted(B.sample(5, rng)[:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0], "dairesel: en eskiler silinir")
        B2 = ns["ReplayBuffer"](100)
        first = need(ns, "mixed_fakes")(np.full((10, 1), 0.0), B2, 0.5, rng)
        ensure(np.all(first == 0) and len(B2) == 10, "ilk adımda tampon boş → yalnız güncel")
        for step in range(1, 30):
            out = ns["mixed_fakes"](np.full((10, 1), float(step)), B2, 0.5, rng)
        ensure((out == 29).sum() == 5 and len(B2) == 100, "yarı güncel, yarı geçmiş")
        ages = 29 - out[out != 29]
        R_.note(f"Son adımda ayırıcının sahte batch'i: 5 güncel örnek + tampondan 5 örnek (yaşları {sorted(ages.astype(int).tolist())} adım). Tampon kapasitesi ayırıcının 'hafızasını' belirler; çok büyük tampon, üreticinin artık üretmediği örneklere karşı ayrım öğretir (gecikme), çok küçük tampon etkisizdir.")
    R_.test("deneyim tekrarı", t)
