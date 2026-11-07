NumPy, döngüsüz.

- `im2col(img, K)` → `(H·W, K²)` (sıfır dolgu, `sliding_window_view`).
- `conv_im2col(img, k)` — im2col @ k.ravel() (fp32).
- `conv_multi(x, w)` — `x (C_in, H, W)`, `w (C_out, C_in, K, K)` → `(C_out, H, W)`: `w.reshape(C_out, -1) @ cols`, cols `(C_in K², H W)`.
- `conv_fft(img, k)` — korelasyon: görüntü ve (ters çevrilmiş) çekirdek $(H+K-1)\times(W+K-1)$'e sıfır dolgu, `rfft2`, çarp, `irfft2`, ortadaki H×W.
- `tile_intensity(BM, BN, BK)` — bir faz için FLOP/bayt (fp32).
