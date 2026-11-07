NumPy, döngüsüz (Elo yinelemesi hariç).

- `symmetry_perms()` → `(8, 9)` int: `np.arange(9).reshape(3,3)`'ün 4 döndürmesi ve yansımaları (`np.rot90`, `np.fliplr`); her satır, dönüşmüş tahtanın i. karesinin eski indeksi.
- `augment(X, PI, Z)` — `X (N, 18)` (iki 9'luk düzlem), `PI (N, 9)`, `Z (N,)` → 8N örnek (gelişmiş indeksleme).
- `az_loss(z, v, pi, logits, theta_sq, c=1e-4)` — ortalama $(z-v)^2-\sum\pi\log\mathrm{softmax}(\text{logits})+c\,\|\theta\|^2$ (logsumexp kararlı).
- `elo_expected(ra, rb)`; `fit_elo(Wm, n_iter=500)` — `Wm[i, j]` i'nin j'ye karşı skoru (beraberlik ½); Bradley–Terry MM (Hunter 2004): $\gamma_i\leftarrow W_i/\sum_{j\ne i}n_{ij}/(\gamma_i+\gamma_j)$ (köşegen yok sayılır); derece $400\log_{10}\gamma$, ortalaması 0.
