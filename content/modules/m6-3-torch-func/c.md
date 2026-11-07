NumPy, lojistik regresyon $p=\sigma(Xw)$, $\ \ell=\frac1N\sum_i\text{BCE}(p_i,y_i)$. **Döngü yok.**

- `per_sample_grads(w, X, y)` → `(N, d)`: $(p_i-y_i)x_i$.
- `hessian(w, X)` → $\frac1NX^\top\operatorname{diag}(p(1-p))X$.
- `hvp(w, X, v)` → $Hv$ ama $H$'yi kurmadan: $\frac1NX^\top\big(p(1-p)\odot Xv\big)$.
