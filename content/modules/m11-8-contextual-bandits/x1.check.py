import numpy as np
from labcheck import close, need, ensure, fmt_time


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        A = rng.normal(0, 1, (4, 4))
        A = A @ A.T + np.eye(4)
        x = rng.normal(0, 1, 4)
        close(need(ns, "sm_update")(np.linalg.inv(A), x), np.linalg.inv(A + np.outer(x, x)), atol=1e-10, what="Sherman–Morrison")
        rows = []
        for d in (50, 200, 400):
            ti, ts, err = need(ns, "time_compare")(d, 30)
            ensure(err < 1e-8, "aynı sonuç")
            rows.append([d, fmt_time(ti / 30), fmt_time(ts / 30), f"{ti / ts:.1f}×", f"{err:.1e}"])
        R_.table("Güncelleme başına süre", ["d", "tam ters O(d³)", "Sherman–Morrison O(d²)", "hızlanma", "max fark"], rows,
                 note="Uzun koşularda yuvarlama hatası birikebilir: pratikte arada bir Cholesky ile yeniden hesaplanır.")
    R_.test("Sherman–Morrison", t)
