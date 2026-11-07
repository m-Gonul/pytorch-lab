Saf Python; görüntü ve çekirdek liste listesi, çekirdek boyutları tek.

- `correlate2d(img, k)` ve `convolve2d(img, k)` — sıfır doldurma, çıkış girişle aynı boyutta.
- `box_filter(img, size)` — `size × size` ortalama, sıfır doldurma (kenarda da `size²`'ye böl).

Kontrol `cv2.filter2D(…, borderType=cv2.BORDER_CONSTANT)` ve `cv2.boxFilter(…, normalize=True, borderType=cv2.BORDER_CONSTANT)` ile float32'de karşılaştırır. (Alıştırmadaki gibi `cv2.blur`'u varsayılan kenarıyla kullanırsan kenarlar farklı çıkar: varsayılan `REFLECT_101`.)
