`load_inline` ile C++ operatörü:

- `CPP_SRC` — `torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor b)`: `x` `(n, d)` fp32 bitişik, `b` `(d,)`; $y=\operatorname{gelu}_{\tanh}(x+b)$. `TORCH_CHECK` ile boyut, dtype ve bitişikliği doğrula. Ham işaretçilerle iki döngü.
- `build()` → derlenmiş modül (`load_inline(name=…, cpp_sources=CPP_SRC, functions=["bias_gelu"])`). İlk derleme ~20 sn sürer, sonra önbellekten gelir.

GPU varsa (isteğe bağlı) `CUDA_SRC` içinde aynı işi yapan bir `__global__` çekirdek ve onu çağıran `torch::Tensor bias_gelu_cuda(torch::Tensor x, torch::Tensor b)` yaz (thread başına bir eleman, `blocks = ⌈N/256⌉`). Kontrol yalnız CUDA varken onu `cuda_sources=` ile derleyip sınar.
