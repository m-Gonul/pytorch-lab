NumPy, döngüsüz.

- `patchify(x, P)` — `(C, H, W)` → `(N, P·P·C)`; yamalar row-major, yama içi sıra `(p1, p2, c)` (einops: `c (h p1) (w p2) -> (h w) (p1 p2 c)`).
- `unpatchify(z, P, C, H, W)` — tersi.
- `patch_embed_conv(x, Wc, b, P)` — `Wc (D, C, P, P)`, adım P'li konvolüsyon (`sliding_window_view` + `einsum`) → `(N, D)`; `patchify(x) @ E + b` ile aynı olmalı (E'yi Wc'den yeniden düzenleyerek).
- `conv_to_linear(Wc)` → `(P·P·C, D)`.
- `mean_attention_distance(A, g, P)` — `A (N, N)` yamalar arası attention (satırlar 1'e toplanır), `g × g` ızgara: her sorgu için piksel cinsinden ağırlıklı ortalama uzaklık, sonra sorgular üzerinden ortalama (makale Şekil 7).
