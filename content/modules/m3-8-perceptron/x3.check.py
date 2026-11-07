import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        db, dW = need(ns, "bias_vs_weight_grads")([0.0, 1e-4, 0.9], [0.5, -0.2])
        close(db, np.array([0.5, -0.2]), what="bias gradyanı = δ")
        close(dW, np.outer([0.5, -0.2], [0.0, 1e-4, 0.9]), what="ağırlık gradyanı = δ aᵀ")
        zs = need(ns, "zigzag_fraction")(100000, "sigmoid", 1)
        zt = ns["zigzag_fraction"](100000, "tanh", 1)
        ensure(zs == 1.0 and abs(zt - 0.5) < 0.02, "sigmoid girdilerinde tüm güncellemeler aynı işaretli")
        close(need(ns, "derivative_max")("sigmoid"), 0.25, atol=1e-6, what="σ' ≤ 1/4")
        close(ns["derivative_max"]("tanh"), 1.0, atol=1e-6, what="tanh' ≤ 1")
        R_.table("Ders alıştırması 2", ["soru", "cevap"],
                 [["bias ile ağırlık güncellemesinin farkı", "∂C/∂b = δ, ∂C/∂w = a^{l−1}·δ: ağırlık güncellemesi önceki aktivasyonla ölçeklenir"],
                  ["önceki aktivasyon ≈ 0 ise", "o ağırlık 'yavaş öğrenir' (ölü/doygun nöronların bağlantıları)"],
                  ["sigmoid eksikleri", f"sıfır merkezli değil (güncellemelerin %{100 * zs:.0f}'i aynı işaretli → zikzak), türevi en çok 0.25 (katman başına ≥4× sönüm), doygunluk"],
                  ["tanh eksiği", "sıfır merkezli ama yine doyar: |z| büyükken türev ≈ 0"]],
                 note="Sigmoid çıktıları hep pozitif olduğundan bir nörona gelen tüm ağırlıkların gradyanları aynı işareti taşır: ağırlık vektörü yalnız 'hepsi artar' ya da 'hepsi azalır' yönünde gidebilir.")
    R_.test("aktivasyonlar ve geri yayılım", t)
