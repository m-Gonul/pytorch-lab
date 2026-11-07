NumPy, **döngü yok**.

- `block_match_np(prev, cur, bs, R)` — b ile aynı sonuç, `(nby, nbx, 2)` `int` dizi. `prev`'i `R` ile **büyük bir değerle** (ör. `1e9`'luk `float`) doldurup `sliding_window_view` ile tüm adayların bloklarını, `cur` bloklarını yeniden şekillendir, SAD hacmi `(nby, nbx, (2R+1)², bs, bs)` → toplam → `argmin` (görüntü dışına taşan adaylar dev SAD alır, böylece seçilmez).
- `lk_flow(I0, I1, win=7, thr=1e-2)` — tek seviyeli, tek adımlı LK: $I_x, I_y$ merkezi fark / 2 (`np.gradient`), $I_t=I_1-I_0$, pencere toplamları `scipy.ndimage.uniform_filter(…, win, mode="nearest")` (ortalama da olur, oran değişmez), $\lambda_{min}>$ `thr` olan yerlerde 2 × 2 çözüm, diğer yerlerde 0. `(u, v)`.
