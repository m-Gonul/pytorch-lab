`gmm_em(Z, mu0, iters=50, reg=1e-6)` — `Z (N, d)` `float64` tensör, `mu0 (K, d)`. Başlangıç: $\pi_k=1/K$, $\Sigma_k=\mathrm{cov}(Z)+\text{reg}\cdot I$ (yansız). `iters` kez E + M (M'de $\Sigma_k$'ye `reg·I` ekle). Log-olasılıkları Cholesky ile ($\log|\Sigma|=2\sum\log\mathrm{diag}L$) ve `logsumexp` ile hesapla. `(mu, cov, w, mean_loglik)` — son değer **son E adımındaki** ortalama log-olabilirlik.

Kontrol, sonucu `GaussianMixture(weights_init, means_init, precisions_init, tol=0, max_iter=iters)` ile karşılaştırır ve uzamış (eliptik) kümelerde GMM'in k-means'i nasıl geçtiğini gösterir.
