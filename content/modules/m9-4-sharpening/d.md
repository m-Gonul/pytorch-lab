Türevlenebilir unsharp mask.

- `gaussian_kernel1d(sigma, ksize)` — `sigma` bir tensör olabilir, gradyan akar.
- `UnsharpMask(ksize=9)` — öğrenilebilir `log_sigma` ve `amount`; `forward(x)` ($(H,W)$, [0,1]) → $x+\text{amount}\,(x-G_\sigma*x)$, ayrılabilir `conv2d`, `reflect` doldurma, kırpma yok.
- `fit_unsharp(x, target, steps=400, lr=0.05)` → `(sigma, amount)`.

Kontrol bilinen $\sigma$ ve kazançla üretilmiş bir hedeften parametreleri geri bulmanı bekler.
