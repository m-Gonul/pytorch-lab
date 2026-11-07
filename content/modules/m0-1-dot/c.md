NumPy'da tek çift için `u @ v` ve `np.linalg.norm(u)` yeter; asıl kazanç **çok çift**te. Satırları vektör olan bir `X` (`n × d`) için tüm çiftlerin kosinüs benzerliği:

$$
\hat X = \frac{X}{\lVert X\rVert_{\text{satır}}},\qquad S=\hat X\hat X^\top\quad (n\times n).
$$

Bölme [[broadcasting]] ile olur: normlar `(n, 1)` shape'inde tutulursa `(n, d)` ile sağdan hizalanır. `keepdims=True` tam bunun için var.

- `cosine_matrix(X)` — `(n, n)` benzerlik matrisi. **Döngü yok.**
- `most_similar(X, i)` — `i`. satıra en benzer *başka* satırın indeksi (kendisi hariç). İpucu: köşegeni `-inf` yap, `argmax` al.
