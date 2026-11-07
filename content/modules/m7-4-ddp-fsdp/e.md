Gerçek kolektiflerle ZeRO-1 (optimizer durumu bölünmüş Adam).

- `flatten(ts)` / `unflatten(flat, like)` — tensör listesini tek düz vektöre ve geri.
- `zero1_worker(rank, world, port, out, steps)`: her adımda
  1. `backward`, gradyanları **tek** düz vektöre topla ve **tek** `all_reduce` (kovalama: çok sayıda küçük çağrı yerine bir büyük çağrı), `world`'e böl;
  2. düz parametre vektörünün kendi parçasını (`world`'e bölünecek şekilde sıfırla doldurulmuş, eşit parçalar) yalnız kendi `m`, `v`'nle Adam ile güncelle;
  3. `dist.all_gather` ile parçaları topla, parametrelere geri yaz.

  Parametreleri ve optimizer durumu eleman sayısını kaydet.
- `run_zero1(world, steps)` → `(parametreler, durum_boyutları)`.

Kontrol tek süreçli `torch.optim.Adam` ile karşılaştırır.
