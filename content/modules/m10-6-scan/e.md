PyTorch, `a, b (B, L)` float32.

- `rec_loop(a, b)` — zaman döngüsü (batch vektörel).
- `rec_scan(a, b)` — log₂L adımlı Kogge–Stone (afin çiftler), döngüsü yalnız seviyeler.
- `rec_cumprod(a, b)` — kapalı form $x_t=P_t\sum_{k\le t}b_k/P_k$, $P_t=\prod_{k\le t}a_k$ (`cumprod`, `cumsum`): tek geçiş ama…
- `compact_scan(x, keep)` — `cumsum` ile konum + `scatter` (maskesiz dağıtım: tutulmayanlar "çöp" konuma).
