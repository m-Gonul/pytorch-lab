from labcheck import need


def check(ns, R_):
    def task():
        compare_chain = need(ns, 'compare_chain')
        import math
        one=compare_chain(64,20,1,0.1,0.9,0)
        assert len(one["ordinary"])==20
        assert max(abs(a-b)for a,b in zip(one["ordinary"],one["cv"]))<1e-12
        same=compare_chain(64,20,4,0.1,0.5,0)
        assert max(abs(a-b)for a,b in zip(same["ordinary"],same["cv"]))<1e-12
        # Constant-return on-policy scalar error contracts by (1-alpha)^2 each episode.
        assert abs(same["ordinary"][9]-(1-0.1)**20)<1e-12
        out=compare_chain(1000,100,4,0.1,0.9,0)
        assert all(math.isfinite(x)and x>=0 for v in out.values()for x in v)
        assert out["cv"][-1]<out["ordinary"][-1]
    R_.test('7.10: kontrol değişkeni ve veri verimliliği', task)
