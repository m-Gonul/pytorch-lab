Tüm öznitelikler için **aynı anda** en iyi kütük ve AdaBoost (PyTorch).

- `best_stumps(F, y, w)` — `F (N, K)` `float64`, `y (N,)` 0/1, `w (N,)`: her sütunu sırala (`argsort(0)`), sıralı ağırlıklarla kümülatif toplamlar (c'deki gibi eşit değerlerde yalnız son konum); `e1 = S_neg + (T_pos − S_pos)` (eşiğin altı yüz), `e2 = S_pos + (T_neg − S_neg)` (eşiğin altı yüz değil); her öznitelik için en küçük hata, eşik (c'deki gibi ardışık değerlerin ortası) ve polarite (`e1 ≤ e2` ise +1). `(err (K,), thr (K,), pol (K,))`.
- `adaboost(F, y, T)` — Viola–Jones AdaBoost; `[(j, thr, pol, alpha), …]`.
- `strong_score(F, H, k=None)` — ilk `k` zayıf sınıflandırıcının $\sum\alpha_th_t$'si ve eşiği $\frac12\sum\alpha_t$: `(score (N,), half)`.

Veri: `skimage.data.lfw_subset()` (100 yüz, 100 yüz olmayan, 25 × 25); öznitelikler her pencerenin ortalama ve std'siyle normalize edilmiş Haar değerleri (hazır yardımcı).
