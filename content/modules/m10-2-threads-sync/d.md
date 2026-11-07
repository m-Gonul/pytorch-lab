C++ (`load_inline`, `std::thread`).

- `histogram(img, nthreads, mode)` — `img` uint8 bitişik; 256 kutulu int64 histogram. `mode`:
  - `"single"` — tek thread;
  - `"atomic"` — `std::atomic<int64_t> bins[256]`, `fetch_add(1, relaxed)`;
  - `"mutex"` — tek `std::mutex`, her piksel için kilitle;
  - `"private"` — her thread kendi `int64_t[256]`'sı, sonda birleştir;
  - `"racy"` — **bilerek yanlış**: `v = bins[x].load(); bins[x].store(v + 1);` (veri yarışı yok — atomik — ama yarış koşulu var).
- `false_sharing(nthreads, iters, padded)` → `(geçen_saniye, sayaçlar)`: her thread kendi sayacını `iters` kez `fetch_add`; `padded` ise sayaçlar `alignas(64)` yapılarda.
- `bounded_buffer(n_items, cap, n_prod, n_cons)` → `(tüketilenlerin_toplamı, gözlenen_en_büyük_doluluk)`: üreticiler 1..n_items sayılarını paylaşır, tüketiciler `std::condition_variable` ile sınırlı kuyruktan çeker; bitişte "zehir hapı" (−1) her tüketiciye.
