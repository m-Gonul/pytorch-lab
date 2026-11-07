C++ `std::thread` (`load_inline`). **Mandelbrot** $H\times W$: piksel (r, c) → $c_0=(-2+3c/(W-1))+i\,(1.5\,r/(H-1))$ (yalnız üst yarı: satır maliyetleri çok dengesiz), `maxit` sınırlı kaçış iterasyonu.

- `SRC` — `std::tuple<torch::Tensor, torch::Tensor> mandelbrot(int64_t H, int64_t W, int64_t maxit, int64_t nthreads, std::string mode)` → (iterasyon sayıları `int32 (H, W)`, thread başına toplam iterasyon `int64 (nthreads,)`). `mode`: `"block"` (ardışık satır blokları), `"cyclic"` (satır r → thread r mod p), `"dynamic"` (`std::atomic<int>` sayaçtan sıradaki satırı al).
- `build()` → modül.
