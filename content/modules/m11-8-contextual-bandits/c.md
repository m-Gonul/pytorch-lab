NumPy.

- `posterior(X, r, lam, sigma)` → `(mu, Sigma)` (kapalı form, `np.linalg.solve`/`inv`).
- `sequential_posterior(X, r, lam, sigma)` — satır satır Sherman–Morrison (tek döngü); aynı sonuç.
- `predictive(Xq, mu, Sigma, sigma)` → `(mean, var)` her sorgu satırı için (gürültü dahil), döngüsüz (`einsum`).
- `joint_features(xs, XA)` — $x_s$ `(ds,)` ve eylem öznitelikleri `(K, da)` → `(K, ds·da)`: dış çarpım (etkileşim) öznitelikleri.
