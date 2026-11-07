NumPy, döngüsüz.

- `scan_blocks(a, p)` — `len(a)` p'ye bölünür; `reshape(p, -1)` → satır `cumsum` + dışlayıcı blok ofsetleri.
- `integral_image(img)` — `cumsum` iki eksende (int64).
- `conv2d(img, k)` — `sliding_window_view` + `einsum`, fp32, sıfır dolgu, uint8 çıktı (b ile aynı yuvarlama).
- `conv2d_separable(img, kx, ky)` — önce yatay, sonra dikey 1B (fp32).
- `matmul_tiled(A, B, T)` — `n` T'ye bölünür; `A.reshape(n/T, T, n/T, T)` ve `einsum` ile karo karo toplam.
