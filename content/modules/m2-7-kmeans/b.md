Saf Python; noktalar ve merkezler tuple/liste.

- `assign(points, centers)` → etiket listesi (0-tabanlı, eşitlikte küçük indeks).
- `update(points, labels, centers)` → yeni merkezler; boş küme eski merkezini korur.
- `kmeans(points, centers, iters)` → `(centers, labels, inertia)`; `iters` kez ata+güncelle, sonra son merkezlere göre son atama ve inertia.
