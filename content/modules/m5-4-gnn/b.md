Saf Python; `edges` yönsüz kenar listesi `[(0, 1), (1, 2)]` (her kenar bir kez), düğümler `0..n-1`, `X` liste listesi, `W` liste listesi (`d_in × d_out`).

- `gcn_weights(n, edges)` → `{(i, j): ağırlık}` sözlüğü; öz-döngüler ve **iki yön** dahil.
- `gcn_layer(X, n, edges, W)` → $\hat AXW$: önce her düğüm için $XW$, sonra her `(i, j)` için `out[i] += w_ij * XW[j]`.
- `mean_neighbors(X, n, edges)` — öz-döngüsüz komşu ortalaması (komşusu yoksa sıfır vektör).
