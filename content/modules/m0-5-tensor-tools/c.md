NumPy'da stride'lar **bayt** cinsindendir (`a.strides`, float64 için eleman başına 8 bayt). `np.lib.stride_tricks.as_strided` istenen shape/stride ile bir view kurar; yanlış stride verilirse bellek dışını okur, bu yüzden güvenli sarmalayıcı `sliding_window_view` tercih edilir.

**Döngü yok:**

- `sliding_windows(x, w)` — `(n-w+1, w)` view; `as_strided` ya da `sliding_window_view` ile. Kontrol, sonucun `x` ile belleği **paylaştığını** (`np.shares_memory`) doğrular.
- `moving_average(x, w)` — pencerelerin ortalaması.
- `is_view_of(a, base)` — `a`, `base`'in belleğini mi gösteriyor? (`np.shares_memory`)
