NumPy, **döngü yok**; `cv2.Sobel` ve `scipy.ndimage.label` serbest.

**`canny_cv(img, lo, hi, l2=False)`** — `cv2.Canny(img, lo, hi, L2gradient=l2)` ile **piksel piksel aynı**. OpenCV'nin ayrıntıları (`canny.cpp`):

1. $d_x, d_y$ = `cv2.Sobel(img, cv2.CV_16S, …, ksize=3, borderType=cv2.BORDER_REPLICATE)` (Gauss **yok**; isteyen önceden bulanıklaştırır).
2. `l2=True`: $m=d_x^2+d_y^2$ ve eşikler kare alınır; değilse $m=|d_x|+|d_y|$. Eşikler `floor` ile tamsayı.
3. NMS tamsayıyla: `TG22 = 13573` ($\tan22.5°\cdot2^{15}$), `ax = |dx|`, `ay = |dy| << 15`, `tg22x = ax*TG22`, `tg67x = tg22x + (ax << 16)`.
   - `ay < tg22x` → yatay: `m > sol` ve `m >= sağ`.
   - `ay > tg67x` → dikey: `m > üst` ve `m >= alt`.
   - yoksa çapraz: `s = -1 if (dx ^ dy) < 0 else 1`; `m > m[y-1, x-s]` ve `m > m[y+1, x+s]`.
   - Görüntünün dışındaki büyüklükler 0. Aday olmak için ayrıca `m > lo`.
4. Güçlü: aday ve `m > hi`. Sonuç: en az bir güçlü piksel içeren 8-bağlı aday bileşenleri, 255.

**`marr_hildreth(img, sigma, thr_frac=0.04)`** — $n$ = en küçük tek $\ge\lceil6\sigma\rceil$; `g = cv2.Laplacian(cv2.GaussianBlur(img.astype(np.float64), (n, n), sigma), cv2.CV_64F)`; b'deki sıfır geçişi kuralı, eşik `thr_frac * g.max()`, sınır 0. `bool` döndür.
