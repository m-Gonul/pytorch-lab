Saf Python.

- `ae_params(sizes, tied=False)` — simetrik yığılmış AE: `sizes` kodlayıcı katman boyutları (ör. `[784, 100, 30]`), kod çözücü aynalanır.
- `kl_sparsity(rho, rho_hat)` ve `kl_sparsity_grad(rho, rho_hat)`.
- `greedy_plan(sizes)` → açgözlü katman katman eğitimin aşamaları: `[(girdi boyu, gizli boyu), …]`.
- `masking_noise(x, p, rng)` (her bileşeni p olasılıkla sıfırla) ve `gaussian_noise(x, sigma, rng)`.
