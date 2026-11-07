import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for s in range(3):
            X, A, R, th = need(ns, "make_logs")(25000, s)
            ensure(X.shape == (25000, 5) and np.allclose(X.sum(1), 1), "simpleks bağlamlar")
            est, cnt = need(ns, "replay_linucb")(X, A, R)
            on = np.mean([need(ns, "online_linucb")(th, cnt, s * 10 + k) for k in range(5)])
            rnd = float(np.mean(R))
            ensure(abs(est - on) < 0.025, f"tekrar oynatma ≈ çevrimiçi ({est:.3f} vs {on:.3f})")
            ensure(abs(cnt - 5000) < 300, "kabul ≈ n/K")
            rows.append([s, cnt, f"{rnd:.3f}", f"{est:.3f}", f"{on:.3f}"])
        R_.table("25 000 rastgele kayıt, 5 eylem", ["tohum", "kabul edilen", "kayıt politikası CTR", "LinUCB (tekrar oynatma)", "LinUCB (çevrimiçi, aynı adım sayısı)"], rows,
                 note="Kayıt politikası düzgün rastgele olduğu sürece tekrar oynatma yansızdır; bedeli veri: olayların ancak 1/K'sı kullanılır.")
    R_.test("tekrar oynatma", t)
