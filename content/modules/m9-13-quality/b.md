Saf Python, düz listeler.

- `mse(x, y)`, `psnr(x, y, R=255)` (MSE = 0 ise `float("inf")`).
- `ssim_global(x, y, L=255, K1=0.01, K2=0.03)` — **tek pencere** (tüm liste); ortalamalar, **yansız** varyans ve kovaryans ($N-1$), birleşik formül.
- `ssim_components(x, y, L=255, K1=0.01, K2=0.03)` → `(l, c, s)`, $C_3=C_2/2$. Çarpımları `ssim_global`'a eşit olmalı.
