NumPy, döngüsüz. Ağırlık düzeni PyTorch'unki: `w (C_in, C_out, k, k)`.

- `conv_matrix(w, H, s=1, p=0)` — tek kanallı 2B konvolüsyonun (çapraz korelasyon, çekirdek `w (k, k)`, `H × H` girdi) matrisi $C$, şekil `(H_o², H²)`: `C @ x.ravel()` = konvolüsyon.
- `tconv2d(x, w, s=1, p=0)` — `x (C_in, H, W)`: scatter (`np.add.at` ile), sonra kırp.
- `tconv_zero_insert(x, w, s=1, p=0)` — sıfır ekle, `k − 1 − p` dolgula, kanal eksenlerini değiştirip çekirdeği 180° çevir, valid korelasyon.
- `bilinear_kernel(f)` — faktör `f` için `k = 2f − f mod 2` boyutlu 2B çekirdek (FCN başlatması).
- `confusion(pred, gt, C)` (`bincount`), `miou(cm)` → (sınıf IoU'ları, ortalama).
