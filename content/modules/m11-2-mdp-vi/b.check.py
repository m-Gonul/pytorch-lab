import numpy as np
from labcheck import close, no_imports, need, ensure


def ref(P, R, VT, h, g):
    P, R = np.array(P), np.array(R)
    V = np.array(VT, float); Vs = [V]; pis = []
    for _ in range(h):
        Q = R + g * np.einsum("ast,t->sa", P, V)
        pis.append(Q.argmax(1)); V = Q.max(1); Vs.append(V)
    return Vs[::-1], pis[::-1]


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def ret():
        P, R, VT = need(ns, "retail_mdp")()
        ensure(np.allclose(np.array(P).sum(-1), 1), "satırlar olasılık")
        Vs, pis = need(ns, "finite_vi")(P, R, VT, 5)
        Ws, qis = ref(P, R, VT, 5, 1.0)
        for t in range(6):
            close(np.array(Vs[t]), Ws[t], atol=1e-9, what=f"V_{t}")
        for t in range(5):
            ensure(list(pis[t])[1:] == list(qis[t])[1:], f"π_{t}")
        close(np.array(Vs[4]), np.array([-9, -1.8, -5.8, -10.8]), atol=1e-9, what="V_4 (notlar)")
        name = lambda a: "yüksek" if a == 0 else "indirim"
        R_.table("Perakende stok, h = 5: en iyi eylem (S1, S2, S3)", ["t (kalan gün)", "S1", "S2", "S3", "V_t(S1)"],
                 [[f"{t} ({5 - t})", name(pis[t][1]), name(pis[t][2]), name(pis[t][3]), f"{Vs[t][1]:.4f}"] for t in range(5)],
                 note="Son günde her stok düzeyinde indirim; daha fazla gün kalınca S1 ve sonra S2'de yüksek fiyata geçiş: durağan olmayan politika.")
    R_.test("perakende problemi (finite_vi)", ret)

    def rnd():
        rng = np.random.default_rng(0)
        P = rng.random((3, 6, 6)); P /= P.sum(-1, keepdims=True); R = rng.normal(size=(6, 3)); VT = rng.normal(size=6)
        Vs, pis = ns["finite_vi"](P.tolist(), R.tolist(), VT.tolist(), 7, 0.9)
        Ws, qis = ref(P, R, VT, 7, 0.9)
        for t in range(8):
            close(np.array(Vs[t]), Ws[t], atol=1e-9, what=f"rastgele MDP V_{t}")
    R_.test("rastgele MDP", rnd)
