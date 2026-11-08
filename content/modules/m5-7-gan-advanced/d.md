PyTorch. Veri: 16 × 16'ya büyütülmüş rakamlar, [−1, 1]. Değerlendirme için verilen küçük bir CNN sınıflandırıcının sondan bir önceki katmanı "Inception" vekili.

- `Generator(z=32, c=64)` — DCGAN: ConvT(z → 2c, 4, 1, 0) → BN → ReLU → ConvT(2c → c, 4, 2, 1) → BN → ReLU → ConvT(c → 1, 4, 2, 1) → tanh. Girdi `(B, z)`.
- `Discriminator(c=64)` — Conv(1 → c, 4, 2, 1) → LeakyReLU(0.2) → Conv(c → 2c, 4, 2, 1) → BN → LeakyReLU → Conv(2c → 1, 4, 1, 0); BN girişte yok. Çıkış logit `(B,)`.
- `train_gan(G, D, X, epochs, z=32, lr=2e-4)` — Adam(β₁ = 0.5); önce ayırıcı adımı, sonra doymayan üretici adımı.
- `fid_np(F1, F2)` — c aşamasındaki FID (torch tensörlerinden numpy'a çevirerek).
