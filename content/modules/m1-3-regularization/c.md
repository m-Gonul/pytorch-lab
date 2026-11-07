NumPy'da bias'lı ridge ve tüm λ'lar için yol.

- `ridge(X, y, lam)` → `(w, b)`. Önce `X` ve `y`'yi ortala, $(X_c^\top X_c+\lambda I)w=X_c^\top y_c$'yi `np.linalg.solve` ile çöz, $b=\bar y-\bar x^\top w$. **Döngü yok.**
- `ridge_path(X, y, lams)` → `(L, d)`; her satır bir λ'nın `w`'si. Ortalanmış $X_c=U\,\text{diag}(s)\,V^\top$ (ince SVD) için

$$
w(\lambda)=V\,\text{diag}\!\left(\frac{s}{s^2+\lambda}\right)U^\top y_c .
$$

SVD bir kez alınır; λ'lar [[broadcasting]] ile tek ifadede: `s / (s**2 + lams[:, None])` → `(L, k)`. **Döngü yok.**
