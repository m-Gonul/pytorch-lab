Saf Python; "thread"ler sıralı döngülerle taklit edilir (önemli olan iş bölümü ve bağımlılıklar).

- `block_scan(a, p)` — iki geçişli kapsayıcı tarama; `p` blok.
- `integral_image(img, p)` — liste listesi; satır geçişi `p` satır grubuna, sütun geçişi `p` sütun grubuna bölünmüş.
- `conv_bands(img, k, p)` — `k` kare katsayı listesi (tek kenar), **sıfır dolgulu "aynı" boyut** korelasyon ($\sum k[u][v]\,f(y+u-r,x+v-r)$, $r=\lfloor K/2\rfloor$), çıktı `min(255, max(0, int(acc + 0.5)))`; satırlar `p` banda bölünür ve her bant yalnız kendi halo'lu girdi dilimini kullanır.
- `matmul_rows(A, B, p)` — C'nin satır blokları, iç çarpım ikj sırasıyla.
