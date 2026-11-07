Saf Python. `X` liste listesi, `y` tamsayı etiketler.

- `gini(labels)`
- `best_split(X, y)` → `(j, t, skor)` ya da bölünecek eşik yoksa `None`. Aday eşikler her özellikte sıralı **farklı** değerlerin orta noktaları. Eşitlikte küçük özellik indeksi, sonra küçük eşik kazanır.
- `build_tree(X, y, max_depth)` — düğüm: yaprak `{"leaf": sınıf}` ya da `{"j": …, "t": …, "left": …, "right": …}`. Saf düğümde, `max_depth == 0`'da ya da bölünme yoksa yaprak; yaprak sınıfı çoğunluk (eşitlikte küçük etiket). Sol: $x_j\le t$.
- `predict_tree(tree, x)`
