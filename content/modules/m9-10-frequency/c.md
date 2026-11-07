NumPy, **döngü yok**.

- `lowpass(kind, shape, D0, n=2)` — merkezli $H$, `shape = (P, Q)`, $D(u,v)=\sqrt{(u-P/2)^2+(v-Q/2)^2}$ (tamsayı merkez `P//2`, `Q//2`); `kind` ∈ `"ideal"`, `"butterworth"`, `"gaussian"`. `highpass = 1 - lowpass` (ayrı fonksiyon).
- `freq_filter(img, H)` — Gonzalez'in 7 adımı; `H` şekli $(2M, 2N)$; `float64` döndür.
- `conv_fft(img, k)` — **tam** doğrusal konvolüsyon, `scipy.signal.convolve2d(img, k, "full")` ile aynı; `np.fft.rfft2(…, s=…)` ile.
