Saf Python, ring all-reduce simülasyonu. `vecs`: $N$ işçinin eşit uzunluklu listeleri (uzunluk $N$'ye bölünür).

- `ring_allreduce(vecs)` → `(sonuçlar, gönderilen)`: `sonuçlar[i]` işçi $i$'nin son vektörü (hepsi eleman toplamı), `gönderilen[i]` işçi $i$'nin gönderdiği eleman sayısı.
  - Reduce-scatter, adım $s=0..N-2$: işçi $i$, parça $(i-s)\bmod N$'yi işçi $(i+1)\bmod N$'ye gönderir; alıcı onu kendi aynı parçasına **ekler**.
  - All-gather, adım $s=0..N-2$: işçi $i$, parça $(i+1-s)\bmod N$'yi gönderir; alıcı **üzerine yazar**.
  - Her adımda tüm mesajları önce hazırla, sonra uygula (eşzamanlı gönderim).
