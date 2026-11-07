Saf Python; görüntü liste listesi. Kenarda `BORDER_REFLECT_101` (cb|abc…, kenar tekrarlanmaz): indeks $-1\to1$, $W\to W-2$.

- `diff1(f)`, `diff2(f)` — 1-boyutlu liste; `diff1` uzunluğu $n-1$, `diff2` uzunluğu $n-2$ ($x=1..n-2$).
- `laplacian(img)` — 4-komşulu çekirdek.
- `sobel(img)` → `(gx, gy)`.

Kontrol `cv2.Laplacian(…, ksize=1)` ve `cv2.Sobel(…, ksize=3)` ile float32'de karşılaştırır.
