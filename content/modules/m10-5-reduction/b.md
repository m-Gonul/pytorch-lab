Saf Python.

- `reduce_tree(xs, op, identity)` → `(sonuç, adım, işlem)`: her turda çiftleri birleştir (tek sayıda ise son elemanı birim ile eşle ama sayma).
- `warp_activity(n_threads, mapping)` → her iterasyon için `(aktif thread, aktif warp, ayrışan warp)` listesi; `mapping` `"simple"` ya da `"convergent"`; 2·n_threads eleman.
- `run_block(x, mapping)` — aynı iki eşlemeyi gerçekten `x` (uzunluk 2·blockDim) üzerinde çalıştır (yerinde toplama), sonucu döndür.
- `hist_private(data, n_threads, n_bins)` → `(histogram, birleştirme_toplamaları)`: her "thread"in kendi kopyası (veri döngüsel dağıtılır), sonra birleştirme.
