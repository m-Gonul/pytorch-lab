NumPy, **döngü yok**, `uint8` gri seviye; `np.lib.stride_tricks.sliding_window_view`. SE `bool` maske (tek boyutlu), orijin merkezde. Kenar kuralı OpenCV'nin varsayılanı: erozyonda dışarısı **255**, genişlemede **0** (yani dışarısı hiç seçilmez).

- `gray_erode(f, se)`, `gray_dilate(f, se)` (ders kitabı: yansıtılmış SE ile max).
- `opening(f, se)`, `closing(f, se)`, `gradient(f, se)`, `tophat(f, se)`, `blackhat(f, se)`.
- `nonflat_erode(f, bN)` — `bN` tamsayı dizisi; `float64` döndür, dışarısı $+\infty$.

Simetrik SE'lerde `cv2.morphologyEx` ile birebir; asimetrikte `cv2.morphologyEx(MORPH_OPEN)` ders kitabından ayrılır (neden?).
