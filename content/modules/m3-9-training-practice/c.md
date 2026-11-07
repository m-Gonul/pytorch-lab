NumPy.

- `kfold_indices(n, k, seed)` → k `(train, val)` çifti (karıştırılmış, katlar eşit ±1).
- `poly_cv(x, y, degrees, k=5, seed=0)` → `{derece: ortalama doğrulama MSE}` — polinom en küçük kareleri (`np.vander` + `lstsq`); döngü katlar ve dereceler üzerinde.
- `early_stop(val_losses, patience)` → `(durma_epoch'u, en_iyi_epoch)` — en iyi sonrası `patience` epoch iyileşme yoksa durulan epoch.
- `standardize_leak(Xtr, Xte)` ve `standardize_ok(Xtr, Xte)` — ilki istatistikleri birleşik veriden, ikincisi yalnız eğitimden.
