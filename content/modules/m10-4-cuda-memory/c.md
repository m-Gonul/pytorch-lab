NumPy, döngüsüz: tüm grid'in warp'larını bir dizide sayar.

- `warp_sectors(addr)` — `addr (n_warps, 32)` bayt adresleri → warp başına sektör sayısı (satır içi sıralama + değişim sayımı).
- `warp_bank_degree(words)` — `(n_warps, 32)` kelime adresleri → çakışma derecesi (benzersiz adresler, `np.add.at` ile bank sayımı).
- `transpose_traffic(N, kind, T=32, pad=0)` → `(okuma sektörleri, yazma sektörleri, paylaşımlı okuma çakışma derecesi ortalaması)`; `kind`: `"copy"`, `"naive"`, `"tiled"`. Thread (y, x) blok içinde x-ana sırada warp'lara bölünür; `"tiled"`: blok T×T, karo `tile[ty][tx]`'e birleştirilmiş okunur, `tile[tx][ty]`'den (kelime adresi `tx·(T + pad) + ty`) okunup birleştirilmiş yazılır. N, T'nin katı, float32.
