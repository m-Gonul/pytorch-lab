import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows, res = [], {}
        for div in (True, False):
            for pol in ("greedy", "ts"):
                R = np.array([need(ns, "linear_bandit")(2000, pol, div, s) for s in range(40)])
                res[(div, pol)] = R
                rows.append(["çeşitli N(0, I)" if div else "neredeyse sabit", pol, f"{R.mean():.1f}", f"{np.median(R):.1f}", f"{R.max():.0f}"])
        ensure(res[(True, "greedy")].mean() < 2 * res[(True, "ts")].mean(), "çeşitli bağlamda açgözlü yeterli")
        ensure(res[(False, "greedy")].max() > 10 * res[(False, "ts")].max(), "sabit bağlamda açgözlü bazı deneylerde takılmalı")
        R_.table("Doğrusal haydut, 2000 adım, 40 deney (pişmanlık)", ["bağlam", "politika", "ortalama", "ortanca", "en kötü"], rows,
                 note="Bağlamlar her yöne yeterince dağılıyorsa açgözlü seçim bile her parametreyi öğrenir (Bastani ve ark. 2021); bağlam sabite yakınsa problem bağlamsız haydutlaşır ve açgözlü takılır.")
    R_.test("bağlam çeşitliliği", t)
