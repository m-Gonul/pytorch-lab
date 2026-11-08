import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        X, v, w = rng.normal(size=(32, 6)), rng.normal(size=32), rng.normal(size=6)
        L, g = need(ns, "loss_and_grad")(w, X, v)
        h = 1e-6
        num = np.array([(ns["loss_and_grad"](w + h * e, X, v)[0] - L) / h for e in np.eye(6)])
        close(g, num, atol=1e-5, what="analitik gradyan")
        close(ns["loss_and_grad"](3.7 * w, X, v)[0], L, atol=1e-12, what="L(c·w) = L(w)")
        close(float(g @ w), 0.0, atol=1e-10, what="∇L ⊥ w")
        rows = []
        for c in (0.25, 1.0, 4.0):
            gc = ns["loss_and_grad"](c * w, X, v)[1]
            rows.append([c, f"{np.linalg.norm(c * w):.2f}", f"{np.linalg.norm(gc):.4f}", f"{np.linalg.norm(gc) / np.linalg.norm(c * w):.4f}"])
        close(float(rows[0][2]) / float(rows[2][2]), 16.0, atol=0.01, what="‖∇‖ ∝ 1/c")
        R_.table("BN öncesi ağırlıkları c ile ölçeklemek", ["c", "‖w‖", "‖∇L‖", "göreli adım ‖∇L‖/‖w‖"], rows,
                 note="Kayıp ölçeğe kayıtsız; gradyan ağırlığa diktir ve 1/‖w‖ ile ölçeklenir: bir SGD adımının açısal etkisi η/‖w‖². Gradyan w'ye dik olduğu için her adım ‖w‖'yi büyütür (Pitagoras) — öğrenme kendiliğinden yavaşlar. Weight decay ‖w‖'yi küçültüp bunu dengeler: BN'li ağlarda L2'nin asıl etkisi etkin öğrenme oranını yüksek tutmaktır.")
    R_.test("ölçek değişmezliği", t)
