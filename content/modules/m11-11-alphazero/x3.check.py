import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "elo_diff_mle")(3, 1), 400 * np.log10(3), what="MLE")
        m, lo, hi = need(ns, "elo_posterior")(4, 1)
        ensure(lo < 0 < m < hi, "4–1 sonucunda belirsizlik büyük: %90 aralık 0'ı kapsar")
        m2, lo2, hi2 = ns["elo_posterior"](100, 0)
        ensure(lo2 > 500, "100–0: alt sınır yüksek")
        R_.table("Maç sonucundan Elo farkı (Beta(1,1) önselli kazanma olasılığı)", ["maç", "MLE", "ortanca", "%90 aralık"],
                 [["AlphaGo Lee – Lee Sedol 4–1", f"{ns['elo_diff_mle'](4, 1):.0f}", f"{m:.0f}", f"[{lo:.0f}, {hi:.0f}]"],
                  ["AlphaGo Zero – AlphaGo Lee 100–0", "∞", f"{m2:.0f}", f"[{lo2:.0f}, {hi2:.0f}]"]],
                 note="Beş oyun güç farkını kabaca ölçer; 100–0'da MLE sonsuzdur ve yalnız alt sınır anlamlıdır. Makaleler bu yüzden Elo'yu birçok oyuncunun birçok oyunundan Bradley–Terry ile (c aşaması) hesaplar.")
    R_.test("Elo belirsizliği", t)
