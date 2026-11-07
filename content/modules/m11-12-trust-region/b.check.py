import math
import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def derivs():
        p = [0.3, -0.7]
        h = 1e-6
        g = need(ns, "rosen_grad")(p)
        num = [(ns["rosen"]([p[0] + h, p[1]]) - ns["rosen"]([p[0] - h, p[1]])) / (2 * h), (ns["rosen"]([p[0], p[1] + h]) - ns["rosen"]([p[0], p[1] - h])) / (2 * h)]
        close(np.array(g), np.array(num), rtol=1e-6, what="gradyan")
        H = np.array(need(ns, "rosen_hess")(p))
        close(H, H.T, what="simetrik Hessian")
    R_.test("türevler", derivs)

    def dl():
        H = [[2.0, 0.0], [0.0, 1.0]]
        close(np.array(need(ns, "dogleg")([2.0, 1.0], H, 10.0)), np.array([-1.0, -1.0]), what="bölge geniş: Newton adımı")
        s = ns["dogleg"]([2.0, 1.0], H, 0.5)
        close(np.linalg.norm(s), 0.5, what="dar bölge: sınırda")
        s = ns["dogleg"]([1.0, 0.0], [[-1.0, 0], [0, 1.0]], 0.3)
        close(np.array(s), np.array([-0.3, 0.0]), what="negatif eğrilik: Cauchy noktası sınırda")
    R_.test("dogleg", dl)

    def cmp():
        tr = need(ns, "trust_region")([-1.2, 1.0], 100)
        gd = need(ns, "gd_backtracking")([-1.2, 1.0], 100)
        dist = lambda p: math.hypot(p[0] - 1, p[1] - 1)
        k_tr = next((i for i, p in enumerate(tr) if dist(p) < 1e-6), None)
        ensure(k_tr is not None and k_tr < 60, "güven bölgesi 60 yinelemede (1,1)'e ulaşmalı")
        ensure(dist(gd[-1]) > 0.05, "gradyan inişi 100 yinelemede hâlâ uzakta")
        gd5 = ns["gd_backtracking"]([-1.2, 1.0], 5000)
        k_gd = next((i for i, p in enumerate(gd5) if dist(p) < 1e-3), None)
        R_.table("Rosenbrock, başlangıç (−1.2, 1)", ["yöntem", "10. yineleme", "50.", "100.", "‖p − (1,1)‖ < 1e−3 için"],
                 [["güven bölgesi (dogleg, tam Hessian)", f"{dist(tr[10]):.2e}", f"{dist(tr[50]):.2e}", f"{dist(tr[100]):.2e}", f"{next(i for i, p in enumerate(tr) if dist(p) < 1e-3)}"],
                  ["gradyan inişi + Armijo", f"{dist(gd[10]):.2e}", f"{dist(gd[50]):.2e}", f"{dist(gd[100]):.2e}", f"{k_gd if k_gd is not None else '> 5000'}"]],
                 note="Muz biçimli vadide gradyan yönü kötü koşullu: doğru araması küçük adımlara mahkûm. İkinci dereceden model eğriliği bilir; güven bölgesi modele ancak doğrulandığı mesafede güvenir.")
    R_.test("güven bölgesi vs doğru araması", cmp)
