NumPy; tanh zinciri $h_{i+1}=\tanh(h_iW_i)$, çıkışa gelen gradyan `gout`.

- `grads_full(x, Ws, gout)` — tüm $h_i$'leri sakla, geri geçiş: $\ \delta=g\odot(1-h_{i+1}^2)$, $\ \nabla W_i=h_i^\top\delta$, $\ g\leftarrow\delta W_i^\top$.
- `grads_ckpt(x, Ws, gout, k)` → `(grads, peak)`. İleri geçişte yalnız segment sonlarını (ve `x`'i) sakla; geri geçişte her segmenti başından yeniden hesapla. `peak` = aynı anda saklı **katman çıktısı** sayısının maksimumu (`x` hariç), b'deki tanımla aynı.

Katmanlar/segmentler üzerinde döngü serbest (en fazla iki seviye).
