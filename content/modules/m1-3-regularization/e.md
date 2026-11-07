λ'yı seçmek için [[cross-validation|k-fold cross-validation]]: veriyi $k$ parçaya böl; her parça bir kez doğrulama, kalanlar eğitim. Her λ için ortalama doğrulama hatası; en düşüğü seç.

Döngü yapısı: `for fold` × `for λ` × (çöz + değerlendir). λ döngüsünü kaldır: her fold'da ridge yolunu (c'deki SVD formülü) **tüm λ'lar için birden** hesapla, doğrulama tahminleri `(n_val, L)` matrisi, MSE'ler `(L,)`.

- `ridge_path_torch(X, y, lams)` → `(W: (L, d), b: (L,))`; bias'lı, SVD ile, **döngü yok**.
- `cv_ridge(X, y, lams, k=5)` → `(mse: (L,), best_lam)`. Fold'lar sırayla: `i`. fold `idx[i::k]` (ön karıştırma yok, deterministik). Fold döngüsü serbest, λ döngüsü yok.

Kontrol, 200 λ için `for λ: solve` ile SVD yolunu yarıştırır.
