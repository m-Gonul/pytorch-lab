Saf Python.

- `point_detect(img, T)` — 8-komşulu Laplacian $|R|\ge T$ → 0/1; kenar satır/sütunları 0.
- `line_detect(img, kind, T)` — `kind`: `"h"` $\begin{bmatrix}-1&-1&-1\\2&2&2\\-1&-1&-1\end{bmatrix}$, `"v"` (devriği), `"d1"` (sol üst → sağ alt) $\begin{bmatrix}2&-1&-1\\-1&2&-1\\-1&-1&2\end{bmatrix}$, `"d2"` (sağ üst → sol alt, yatay aynası). Korelasyon; **pozitif** yanıt $\ge T$ → 1. Kenar 0.
- `kmeans(points, init, max_iter=300)` — `points`, `init`: liste listeleri; Lloyd; atamada eşitlikte küçük indis; boş küme ortalamasını korur; ortalamalar **hiç değişmeyince** dur. `(labels, means)` döndür.
