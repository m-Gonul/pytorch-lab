import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        T, d = 9, 8
        X = rng.normal(size=(T, d))
        W = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3)]
        full = need(ns, "decoder_full")(X, *W, True)
        inc = need(ns, "decoder_incremental")(X, *W)
        close(full, inc, atol=1e-12, what="nedensel tam geçiş = adım adım çözümleme")
        nomask = ns["decoder_full"](X, *W, False)
        diff = np.abs(nomask - inc).max(1)
        ensure(diff[-1] < 1e-12 and diff[0] > 1e-3, "maskesiz: yalnız son konum eşleşir")
        rows = [[t, f"{diff[t]:.2e}"] for t in range(T)]
        R_.table("Maskesiz tam geçiş ile adım adım çözümleme arasındaki fark", ["konum t", "max |fark|"], rows,
                 note="Nedensel maskeyle T konumun tamamı tek bir matris çarpımıyla hesaplanır ve her konum, çıkarımda göreceği şeyin aynısını görür: öğretmen zorlaması paralel olur (RNN'de T sıralı adım). Maske olmasaydı eğitimde t. konum geleceği görürdü — 'hile' — ve çıkarımla tutarsız olurdu; tek istisna her şeyi zaten gören son konumdur.")
    R_.test("paralel eğitim", t)
