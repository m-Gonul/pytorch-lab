from labcheck import need


def check(ns, R_):
    def task():
        episodes19 = need(ns, 'episodes19')
        compare_nstep = need(ns, 'compare_nstep')
        import math
        paths = episodes19(10, 0)
        assert len(paths) == 10 and all(p[0] == 10 and p[-1] in (0, 20) for p in paths)
        assert all(abs(a-b) == 1 for p in paths for a,b in zip(p,p[1:]))
        one = compare_nstep(paths, 1, 0.1)
        assert len(one["target"]["V"]) == 21 and len(one["delta"]["rms"]) == 10
        assert max(abs(a-b) for a,b in zip(one["target"]["V"],one["delta"]["V"])) < 1e-12
        out = compare_nstep(paths, 4, 0.4)
        # Independent direct return reference; update order is tau=0..T-1.
        V = [0.0]*21
        for path in paths:
            T=len(path)-1
            for tau in range(T):
                h=min(tau+4,T);G=(1 if path[-1] else -1) if h==T else V[path[h]]
                V[path[tau]] += 0.4*(G-V[path[tau]])
        assert max(abs(a-b) for a,b in zip(V,out["target"]["V"])) < 1e-12
        assert max(abs(a-b) for a,b in zip(out["target"]["V"],out["delta"]["V"])) > 1e-5
        assert all(math.isfinite(x) for z in out.values() for x in z["rms"])
        assert all(z["V"][0] == z["V"][20] == 0 for z in out.values())
    R_.test('7.2: aynı veri ve değişen tablo', task)
