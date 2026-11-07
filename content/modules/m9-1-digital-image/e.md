`warp_batch(imgs, A, t)` — `imgs: (B, H, W)`, `A: (B, 2, 2)`, `t: (B, 2)`; her görüntüye kendi dönüşümü, **tek** `grid_sample` çağrısıyla. Döngü yok.

Kontrol, `d`'deki tek görüntülük fonksiyonun döngüyle çağrılmasıyla karşılaştırır ve süreyi ölçer.
