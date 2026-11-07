Tablodaki tüm normalizasyonlar tek bir fonksiyon: "şu eksenlerde ortalama ve varyans al".

- `norm_over(x, dims, eps=1e-5)` — `x`'i `dims` eksenlerinde normalize et (yanlı varyans), $\gamma/\beta$ yok.
- `group_norm(x, G, gamma, beta, eps=1e-5)` — `x: (N, C, H, W)`; `(N, G, C//G, H, W)`'ye reshape, `norm_over` ile `(2, 3, 4)` eksenlerinde normalize, geri reshape, kanal başına $\gamma,\beta$. **Döngü yok.**

Kontrol: `group_norm` = `F.group_norm`; `G=C` → `F.instance_norm`; `G=1` → `F.layer_norm` (C, H, W üzerinde); `norm_over(x, (0, 2, 3))` → BatchNorm'un eğitim normalizasyonu. Hız tablosu: gruplar üzerinde döngü vs reshape.
