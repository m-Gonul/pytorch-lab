Saf Python, liste listeleri (gri seviye).

- `block_match(prev, cur, bs, R)` — `cur`'un her `bs × bs` bloğu (satır sırası, artakalan kenar atlanır) için önceki karede `dy, dx ∈ [-R, R]` (dış döngü `dy`, iç `dx`), aday blok tamamen görüntü içindeyse SAD; en küçük SAD (eşitlikte ilk bulunan). `[[(dy, dx), …], …]` blok ızgarası.
- `compensate(prev, mvs, bs)` — her bloğu `prev`'in kaydırılmış bloğundan kopyalayarak kestirim karesi; kapsanmayan kenar pikselleri `prev`'den.
- `sad(a, b)` yardımcısı.
