`canny_torch(imgs, lo, hi, l2=False)` — `imgs (B, H, W)` uint8 tensör → `(B, H, W)` uint8 (0/255), her görüntü için `cv2.Canny` ile **birebir** aynı.

- Sobel: c'deki gibi tek `conv2d`, replicate dolgu, `float` hesaplayıp `round().long()` ile tamsayıya geç (değerler küçük, kesin).
- NMS'i `int64` tensörlerle ve kaydırılmış görünümlerle (`F.pad` ile sıfır dolgu) yaz; bit kaydırma `<<` tensörlerde de çalışır.
- Histerezis: bileşen etiketleme yok; `edges ← aday ∧ maxpool3×3(edges)` adımını değişiklik kalmayana kadar tekrarla (tek `while` döngüsü serbest). Kaç tekrar sürdüğüne bak: en uzun kenar zincirinin uzunluğu kadar.
