Saf Python. `op` iki argümanlı fonksiyon.

- `inclusive_scan(x, op)`, `exclusive_scan(x, op, identity)` — sıralı.
- `kogge_stone(x, op)` → `(sonuç, adım, işlem)` — çift tamponlu.
- `kogge_stone_inplace(x, order)` — **tek dizide**, her adımda "thread"ler `order` sırasıyla (`"asc"`: i artan, `"desc"`: azalan) `a[i] = a[i - s] + a[i]` yapar; hangisi doğru?
- `brent_kung(x, op)` → `(sonuç, adım, işlem)` — n 2'nin kuvveti; yukarı süpürme ($s=1,2,\dots,n/2$: $i=2s-1,4s-1,\dots$ için $a_i=a_{i-s}✦a_i$), aşağı süpürme ($s=n/4,\dots,1$: $j=2s-1,4s-1,\dots$, $j+s<n$ için $a_{j+s}=a_j✦a_{j+s}$).
- `hierarchical_scan(x, section, op)` — bölüm taramaları + bölüm toplamlarının taraması + ofset.
