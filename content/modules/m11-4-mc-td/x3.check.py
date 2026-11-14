import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        best = {}
        for n in (1, 2, 4, 8, 16):
            row = [n]
            for a in (0.1, 0.2, 0.4, 0.8):
                e = np.mean([need(ns, "nstep_td")(need(ns, "episodes19")(10, s), n, a) for s in range(20)])
                row.append(f"{e:.3f}"); best[(n, a)] = e
            rows.append(row)
        n_best = min(best, key=best.get)
        ensure(1 < n_best[0] < 16, f"en iyi n arada olmalı, bulunan {n_best}")
        R_.table("19 durumlu yürüyüş: 10. bölüm sonunda 20 tekrar ortalama RMS", ["n", "α = 0.1", "α = 0.2", "α = 0.4", "α = 0.8"], rows,
                 note=f"En iyi (n, α) = {n_best}: bu koşuda ara n iyi;Şekil7.2 ayrıca ilk10 bölümün ortalamasını kullanır.")
    R_.test("n ve α taraması", t)

    def domain():
        walk_values = need(ns, "walk_values")
        nstep_td = need(ns, "nstep_td")
        episodes19 = need(ns, "episodes19")
        # N and gamma are meaningful domain arguments, not just storage sizes.
        for size in (5, 19):
            z=walk_values(size,-1,1)
            assert max(abs(v-(2*(i+1)/(size+1)-1))for i,v in enumerate(z))<1e-12
        assert walk_values(5,-1,0)==[-0.5,0.0,0.0,0.0,0.5]
        a=nstep_td(episodes19(10,0,5,-1),4,.2,5,1,-1,0,True)
        b=nstep_td(episodes19(10,0,5,0),4,.2,5,1,0,.5,True)
        assert len(a)==len(b)==10 and max(abs(x-2*y)for x,y in zip(a,b))<1e-12
    R_.test("N/γ alanı ve ödül dönüşümü", domain)
