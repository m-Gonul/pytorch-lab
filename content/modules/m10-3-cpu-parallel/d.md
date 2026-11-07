C++ `std::thread`.

- `integral_array(x, nthreads)` — `x` int32 → int64 kapsayıcı tarama (iki geçiş; `nthreads = 1` sıralı).
- `integral_image(img, nthreads)` — uint8 → int64 `(H, W)`; satır geçişi ve sütun geçişi (her thread bir sütun aralığı, satırlarda yürür).
- `conv2d(img, k, nthreads)` — uint8 `(H, W)`, fp32 `(K, K)` → uint8, sıfır dolgu, `std::lround` değil **`(int)(acc + 0.5f)`** ve kırpma (b/c ile aynı).
- `matmul(A, B, nthreads, mode)` — fp32; `mode`: `"ijk"`, `"ikj"`, `"bt"` (B devrik kopyası, iç çarpım), `"tiled"` (64'lük karolar, ikj).
