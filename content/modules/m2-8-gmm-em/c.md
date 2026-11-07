$d$ boyut, tam kovaryans, $K$ bileşen; **döngü yok** (iterasyon döngüsü hariç).

Log-yoğunluk tüm $(i,k)$ çiftleri için:

$$
\log\mathcal N_{ik}=-\tfrac12\Big(d\log2\pi+\log\det\Sigma_k+(x_i-\mu_k)^\top\Sigma_k^{-1}(x_i-\mu_k)\Big).
$$

`diff = X[:, None] - mus[None]` `(n, K, d)`, `P = np.linalg.inv(covs)` `(K, d, d)`, Mahalanobis `np.einsum("nkd,kde,nke->nk", diff, P, diff)`, `np.linalg.slogdet(covs)[1]` `(K,)`.

- `log_gauss(X, mus, covs)` → `(n, K)`.
- `e_step_np(X, pis, mus, covs)` → `(R: (n, K), ll: float)`.
- `m_step_np(X, R, reg=1e-6)` → `(pis, mus, covs)`; kovaryans `np.einsum("nk,nkd,nke->kde", R, diff, diff) / Nk[:, None, None]` + `reg·I`.
- `em_np(X, pis, mus, covs, iters)` → `(pis, mus, covs, lls)`.
