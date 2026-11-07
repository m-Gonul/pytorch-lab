Bulanıklık + gürültü: $y=h\circledast x+\eta$ (dairesel, böylece $H$ DFT'de tam köşegen), $\sigma_\eta=0.01$.

- `gaussian_otf(N, s)` — $N\times N$ görüntü, $\sigma=s$ piksellik dairesel Gauss PSF'nin `rfft2` düzenindeki OTF'si: $e^{-2\pi^2s^2(u^2+v^2)/N^2}$, $u$ = `fftfreq(N)·N`, $v$ = `rfftfreq(N)·N`. `(N, N//2+1)`.
- `degrade(x, H, sigma, g)` — `irfft2(rfft2(x)·H) + sigma·randn` (`g` üreteciyle).
- `apply(G, y)` — `irfft2(rfft2(y)·G)`.
- `wiener(H, K)` — $\frac{H}{H^2+K}$ ($H$ gerçel).
- `learn_filter(train, H, sigma, steps=600, batch=16, seed=0)` — 1'lerle başlayan gerçel `G` (`H` ile aynı şekil); her adımda rastgele `batch` görüntü, `degrade`, MSE(`apply(G, y)`, `x`), Adam (lr = 1e-2). `G`'yi döndür.

Doğrusal bir onarıcının MSE'yi en aza indiren hali Wiener'dir; öğrenilen `G` ona (hatta gerçek spektrumu bildiği için sabit-$K$ Wiener'in biraz ötesine) yakınsar.
