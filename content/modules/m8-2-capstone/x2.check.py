import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        rows, wrong = [], {}
        for U in (1, 5, 20):
            x = need(ns, "repaint")(-0.6, 1500, rng, U)
            ensure(np.allclose(x[:, 0], -0.6), "bilinen koordinat korunur")
            wrong[U] = float((x[:, 1] > 0).mean())
            rows.append([U, 200 * U, f"{wrong[U]:.3f}"])
        p_true = 1 / (1 + np.exp(1.2 ** 2 / (2 * 0.09)))
        rows.append(["gerçek koşullu", "—", f"{p_true:.4f}"])
        ensure(wrong[1] > 0.2 and wrong[20] < wrong[1], f"naif yerleştirme başarısız, yeniden örnekleme iyileştirir {wrong}")
        R_.table("x = −0.6 bilinirken y'nin yanlış moda (y > 0) düşme oranı (kâhin ε!)", ["yeniden örnekleme U", "ağ değerlendirmesi", "yanlış mod oranı"], rows,
                 note="Model kusursuz olsa bile naif 'bilineni yerleştir' yöntemi koşullu dağılımdan örneklemez: erken (gürültülü) adımlarda bilinen koordinatın gürültülü sürümü bilinmeyen koordinata neredeyse hiç bilgi taşımaz ve y kendi başına bir moda karar verir; sonradan düzeltilemez. RePaint'in yeniden örneklemesi (gürültüle–tekrar çöz) bilgiyi ileri geri yayarak yardımcı olur ama hesap pahalıdır ve hâlâ kesin değildir. Kesin çözüm için posteriora göre yönlendirme (DPS, ΠGDM) ya da koşullu eğitim gerekir — tıbbi görüntü tamamlamada bu fark klinik olarak önemlidir.")
    R_.test("difüzyonla tamamlama", t)
