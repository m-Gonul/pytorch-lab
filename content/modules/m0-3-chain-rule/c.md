Sayısal gradyanın $n$ koordinat için $2n$ değerlendirme istediğini gördün. Bunları döngü yerine **tek bir batch** olarak yapabiliriz: $x$'in $2n$ kopyasını bir matrise dizip fonksiyonu bir kez çağırırız.

$$
X=\begin{bmatrix} x + hI \\ x - hI\end{bmatrix}\in\mathbb{R}^{2n\times n},\qquad
\nabla f \approx \frac{f(X_{1:n}) - f(X_{n+1:2n})}{2h}
$$

Bu aşamadaki fonksiyonlar satır satır çalışır: `f(X)` `(k, n) → (k,)`, `g(X)` `(k, n) → (k, m)`.

- `num_grad_np(f, x, h=1e-5)` → `(n,)`
- `num_jacobian_np(g, x, h=1e-5)` → `(m, n)`; aynı hile, sonuç transpoze edilmeli.

**Döngü yok.** `np.eye`, `np.vstack`/`np.concatenate` yeterli.
