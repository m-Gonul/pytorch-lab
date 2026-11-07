Hangi (lr, β) çifti? 6 lr × 8 β = 48 kombinasyon. M1.1'deki gibi hepsini **tek tensör** olarak eğit: ağırlıklar `W: (P, d)`, hızlar `V: (P, d)`, her kombinasyonun kendi `lr` ve `beta`'sı `(P,)` vektörlerinde. Tam batch (gürültüsüz), böylece sonuç deterministik.

- `sweep(X, y, lrs, betas, steps)` → son kayıplar `(P,)`; `lrs` ve `betas` aynı uzunlukta `(P,)` (ızgara önceden düzleştirilmiş). Yalnız adım döngüsü; kombinasyonlar üzerinde döngü yok. Bias yok (veri ortalanmış).

Kontrol sonucu tek tek eğitilmiş modellerle karşılaştırır, en iyi çifti ve hız kazancını raporlar.
