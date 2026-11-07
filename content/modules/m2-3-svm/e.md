Kernel SVM'i primal'de, $\alpha$ üzerinden eğit: $K_{ij}=k(x_i,x_j)$, $f=K\alpha+b$,

$$
L(\alpha,b)=\frac1n\sum_i\max(0,1-y_if_i)+\lambda\,\alpha^\top K\alpha .
$$

- `rbf_kernel(A, B, gamma)` → `(n_A, n_B)`; mesafe açılımıyla (M2.1), **döngü yok**.
- `fit_kernel_svm(X, y, gamma, lam=1e-3, steps=300, lr=0.5)` → `(alpha, b)`; tam batch alt-gradyan, yalnız adım döngüsü. $\nabla_\alpha=\frac1nK^\top(-y\odot a)+2\lambda K\alpha$, $a$ aktiflik maskesi.
- `predict_kernel(Xtr, alpha, b, gamma, X)` → işaretler $\pm1$.

Kontrol: iç içe iki çember (lineer olarak ayrılamaz) üzerinde kernel SVM ≥ 0.95 doğruluk alırken lineer SVM şans seviyesinde kalır.
