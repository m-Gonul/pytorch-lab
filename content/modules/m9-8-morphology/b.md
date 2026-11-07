Saf Python, ikili (0/1) liste listeleri. SE: tek boyutlu kare liste, orijin merkezde; 1 = eleman. Görüntü dışı **arka plan** (0).

- `erode(A, B)` — $A\ominus B$.
- `dilate(A, B)` — **ders kitabı** tanımı $A\oplus B=\{z\mid(\hat B)_z\cap A\ne\varnothing\}$, yani $z$'de sonuç 1 ⇔ bir $b\in B$ için $A(z-b)=1$.
- `hit_or_miss(I, B)` — `B` elemanları 1 (ön plan olmalı), 0 (arka plan olmalı), −1 (umursamaz).
- `boundary(A)` — $A-(A\ominus B)$, $B$ = 3 × 3 kare.

Kontrol, `cv2.dilate`'in çekirdeği **yansıtmadan** max aldığını gösterir: asimetrik SE'de seninki yalnız ters çevrilmiş çekirdekle eşleşir.
