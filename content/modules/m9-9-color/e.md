**Vektör median:** penceredeki renk vektörleri $\{c_i\}$ arasından $\sum_j\lVert c_i-c_j\rVert_2$'yi en küçük yapan $c_i$ (eşitlikte ilk). Çıkış her zaman penceredeki renklerden biridir.

`vector_median(img, k)` — `img (H, W, 3)` uint8 tensör → aynı tip. Kenar `replicate`. `F.unfold` → `(H·W, k², 3)`, `torch.cdist` → `(H·W, k², k²)`, satır toplamı, `argmin`, `gather`. Bellek $HW\cdot k^4$: büyük görüntüde satır bloklarına bölmek serbest (tek bir döngü).
