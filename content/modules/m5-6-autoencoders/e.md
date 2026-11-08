16 × 16'ya büyütülmüş rakamlar.

- `ConvAE(code=16)` — Conv(1→16, 3, s2) → Conv(16→32, 3, s2) → düzleştir (32·4·4) → doğrusal → kod; tersi ConvTranspose ile.
- `DenseAE(code=16)` — 256 → 128 → kod → 128 → 256.
- Karşılaştırma: parametre, test MSE, eğitim süresi.
