Saf Python simülasyonu. Katmanlar $0..n-1$, katman $i$'nin çıktı boyutu `sizes[i]`. Her $k$ katmanda bir **segment**; segmentin son katmanının çıktısı saklanır (segment sonu).

- `checkpoint_cost(sizes, k)` → `(tepe, yeniden_hesap)`:
  - Geri geçiş segmentleri **sondan başa** işler. Segment $j$ işlenirken bellekte: segment $0..j$'nin sonları + segment $j$'nin yeniden hesaplanan diğer çıktıları.
  - `tepe` = bunun $j$ üzerinden maksimumu; `yeniden_hesap` = tüm segmentlerde (uzunluk − 1) toplamı.
  - $k=1$ checkpoint'siz durumdur: tepe = `sum(sizes)`, yeniden hesap 0.
- `best_k(sizes)` → tepe belleği en küçük $k$ (eşitlikte küçük $k$).
