İki parça, ikisi de **döngüsüz**.

**Gaussian NB (NumPy).** Her sınıf $c$ ve özellik $j$ için $\mu_{cj}$ ve $\sigma^2_{cj}$ (MLE). Sınıf istatistiklerini döngü yerine one-hot matrisle al: `Y = np.eye(C)[y]` `(n, C)`, sayımlar `Y.sum(0)`, ortalamalar `Y.T @ X / sayım[:, None]`.

- `fit_gnb(X, y, C, eps=1e-9)` → `(log_prior: (C,), mu: (C, d), var: (C, d))`; varyansa `eps` ekle.
- `predict_log_proba_gnb(model, X)` → `(n, C)` normalize log-posterior. Log-yoğunluk için `X[:, None, :]` ile `mu[None]` [[broadcasting]] ile `(n, C, d)`.

**Olasılık tablosu (pandas).**

- `cond_table(df, feature, target, alpha=1.0)` → satırları `target` sınıfları, sütunları `feature` değerleri olan $P(\text{değer}\mid\text{sınıf})$ DataFrame'i (Laplace'lı). `pd.crosstab` tek satır.
