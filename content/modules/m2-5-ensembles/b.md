Saf Python, tek özellik, MSE ile gradient boosting.

- `fit_stump(xs, rs)` → `(t, sol_değer, sağ_değer)`; SSE'yi en çok azaltan orta-nokta eşiği, yapraklar residual ortalaması. Eşitlikte küçük eşik. Hiç bölünme yoksa `(None, ort, ort)`.
- `gbm_fit(xs, ys, rounds, lr)` → model (yapısı serbest).
- `gbm_predict(model, x)` → tek bir tahmin.
