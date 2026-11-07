EM yerel maksimuma takılır; çare k-means'teki gibi çoklu başlatma. $R$ başlatmayı bir batch boyutu yap. Hesabı basit tutmak için **diyagonal** kovaryans: $\log\mathcal N_{rik}=-\frac12\sum_j\big(\log2\pi\sigma^2_{rkj}+\frac{(x_{ij}-\mu_{rkj})^2}{\sigma^2_{rkj}}\big)$, tensör `(R, n, K)`.

- `em_restarts_diag(X, mus0, iters, reg=1e-6)` — `mus0: (R, K, d)`; başlangıçta `pis` eşit, varyanslar her başlatmada `X.var(0, unbiased=False)`. → `(lls: (R,), best: (pis, mus, vars))`. Yalnız iterasyon döngüsü.

Kontrol, her başlatmayı ayrı ayrı koşturan referansla karşılaştırır.
