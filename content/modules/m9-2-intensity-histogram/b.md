Saf Python; görüntü `uint8` değerlerinden oluşan liste listesi.

- `histogram(img)` → 256 elemanlı sayım listesi.
- `equalize_class(img)` — ders formülü, `round((L-1)·c_k/MN)`.
- `equalize_opencv(img)` — $c_{\min}$'li formül; $MN=c_{\min}$ (tek renkli görüntü) ise görüntüyü olduğu gibi döndür.

Kontrol `cv2.equalizeHist` ile birebir karşılaştırır ve iki formülün farkını (ödevdeki "toplam mutlak fark") raporlar.
