Toplu (batch) güncelleme, NumPy, yalnız tekrar döngüsü serbest.

- `batch_mc(episodes, n_states)` — her durum için gözlenen tüm getirilerin ortalaması (ilk ziyaret değil, **her ziyaret**); `episodes` her biri `[(s, r), …]` ve terminale geçiş örtük. Getiri: ters `cumsum` (γ = 1).
- `batch_td(episodes, n_states, alpha=0.01, tol=1e-10)` — tüm geçişleri `(s, r, s2)` dizilerine diz (`s2 = -1` terminal), her taramada artımları **topla** ve birlikte uygula: $\Delta V=\alpha\sum\delta$ (`np.add.at`); değişim < tol olana dek.
- `certainty_equivalence(episodes, n_states)` — sayımlardan $\hat P$, $\hat R$ (terminale geçiş olasılığı dahil), $V=(I-\hat P)^{-1}\hat r$: toplu TD ile aynı olmalı.
