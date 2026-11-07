PyTorch (CPU), `img (H, W)` float32, `k (K, K)`.

- `conv_direct(img, k)` — `F.conv2d`, `padding=K//2`.
- `conv_unfold(img, k)` — `F.unfold` (im2col) + matmul.
- `conv_fft(img, k)` — `torch.fft.rfft2` ile.
- `crossover(H, Ks)` → her K için üç yöntemin süresi ve en büyük farkı.
