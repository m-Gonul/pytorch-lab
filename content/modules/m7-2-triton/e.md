`matmul_kernel(A, B, C, M, N, K, stride…, BM, BN, BK, ACT)` + `matmul(a, b, activation=None)`:

- 2-boyutlu grid `(cdiv(M, BM), cdiv(N, BN))`.
- `acc = tl.zeros((BM, BN), dtype=tl.float32)`; K boyunca `tl.dot(a_tile, b_tile)`; maskeli yüklemelerde `other=0`.
- `ACT: tl.constexpr` — `"relu"` ise `tl.store`'dan önce `tl.maximum(acc, 0)` ([[epilogue füzyonu]]).

Kontrol tek sayılı şekillerle doğrular. CPU yorumlayıcısında süre anlamsız; GPU'da kontrol `torch.matmul` ile karşılaştırma tablosu da basar.
