**Toplu kendine karşı oyun:** G oyun aynı anda; her benzetim adımında her ağaçta seçimle bir yaprağa in, tüm yaprakları **tek** ileri geçişte değerlendir, sonra genişlet ve geri yay. Python ağaç işlemleri sürer, ama ağ çağrısı G kat azalır (GPU'da asıl kazanç budur).

- `encode_batch(boards, players)` → `(n, 18)`.
- `batched_self_play(net, n_games, n_sims, rng, c=1.5)` → d'deki `self_play` ile aynı biçimde örnekler (tüm oyunlar birleşik).
