import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("linear_abar", "cosine_abar", "log_snr", "wasted_fraction")])

    def t():
        L = need(ns, "linear_abar")(1000)
        C = need(ns, "cosine_abar")(1000)
        close(L[1], (1 - 1e-4) * (1 - 1e-4 - (0.02 - 1e-4) / 999), what="doğrusal")
        close(C[499], np.cos(0.508 / 1.008 * np.pi / 2) ** 2 / np.cos(0.008 / 1.008 * np.pi / 2) ** 2, atol=1e-6, what="kosinüs ortası")
        ensure(np.all(np.diff(C) <= 0) and C[-1] < 1e-3, "monoton, sonda ~0")
        wl, wc = need(ns, "wasted_fraction")(L), ns["wasted_fraction"](C)
        ensure(wl > 0.2 and wc < 0.1, f"doğrusal takvim sonda israf ({wl:.2f} vs {wc:.2f})")
        rows = []
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            i = int(q * 1000) - 1
            rows.append([q, f"{L[i]:.4f}", f"{C[i]:.4f}", f"{need(ns, 'log_snr')(L[i:i + 1])[0]:+.2f}", f"{ns['log_snr'](C[i:i + 1])[0]:+.2f}"])
        R_.table("T = 1000: ᾱ ve log SNR", ["t/T", "ᾱ doğrusal", "ᾱ kosinüs", "log SNR doğrusal", "log SNR kosinüs"], rows,
                 note=f"Doğrusal takvim sinyali çok hızlı yok eder: adımların %{wl * 100:.0f}'ında ᾱ < 0.01 (neredeyse saf gürültü — model bu adımlarda az şey öğrenir). Kosinüs takvimi log SNR'yi neredeyse doğrusal düşürür: bilgi yok oluşu zamana eşit yayılır (Nichol & Dhariwal 2021, Şekil 3). Düşük çözünürlüklü görüntülerde bu fark özellikle önemlidir.")
    R_.test("takvimler", t)
