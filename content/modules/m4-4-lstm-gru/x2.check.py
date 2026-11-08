import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "retention")(np.array([0.0, 1.0]), 2), [0.25, (1 / (1 + np.exp(-1))) ** 2], what="σ(b_f)^T")
        bf, bi = need(ns, "chrono_bias")(1000, 100, np.random.default_rng(0))
        ensure(np.all((bf >= 0) & (bf <= np.log(99))) and np.allclose(bi, -bf), "chrono: b_f ∈ [0, log(T_max − 1)], b_i = −b_f")
        rows = []
        curves = {}
        for b in (0.0, 1.0, 3.0):
            curves[b] = need(ns, "memory_curve")(b, 100, 64, 0)
            r = ns["retention"](b, np.array([10, 50, 100]))
            rows.append([b] + [f"{curves[b][k - 1] / curves[b][0]:.1e}" for k in (10, 50, 100)] + [f"{v:.1e}" for v in r])
        ensure(curves[3.0][99] / curves[3.0][0] > 1e6 * curves[0.0][99] / curves[0.0][0], "b_f büyüdükçe bellek uzar")
        R_.table("Rastgele küçük ağırlıklı LSTM (64 birim): ‖∂c_t/∂x₀‖ / ‖∂c₁/∂x₀‖", ["b_f", "t = 10", "t = 50", "t = 100", "σ(b_f)^10", "σ(b_f)^50", "σ(b_f)^100"], rows,
                 note="Başlangıçta ağırlıklar küçükken unutma kapısı ≈ σ(b_f) sabitidir ve bellek tam σ(b_f)^t ile söner — ölçülen eğri formülü izler. b_f = 0 ile 50 adım önceki girdi pratikte görünmezdir: gradyan da aynı çarpanla söner, model uzun bağımlılıkları öğrenmeye hiç başlayamaz. b_f = 1 (Keras/PyTorch'ta unit_forget_bias seçeneği) ya da chrono başlatma (birimlere farklı zaman ölçekleri) bu başlangıç sorununu çözer.")
    R_.test("unutma kapısı bias'ı", t)
