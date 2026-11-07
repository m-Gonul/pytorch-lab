NumPy; tüm örnekler paralel, **yalnız zaman adımları üzerinde tek döngü**.

- `q_sample(x0, t, eps, abar)` — `x0: (n, d)`, `t: (n,)` tamsayı (örnek başına farklı $t$!). Döngü yok.
- `ddpm_sample(eps_fn, betas, n, d, rng)` — `eps_fn(x, t)`, `x: (n, d)`, `t` skaler.
- `ddim_sample(eps_fn, betas, n, d, rng, steps)` — η = 0; zaman adımları `np.linspace(T - 1, 0, steps).round().astype(int)`; son adımdan sonra $\bar\alpha_{\text{önceki}}=1$ (sonuç $\hat x_0$). Rastgelelik yalnız başlangıçta.

Kontrol iki boyutlu Gauss veri için kâhinle iki örnekleyiciyi de sınar ve DDIM'de adım sayısının etkisini tabloya döker. Örnekte $T=300$: $\bar\alpha_T\approx5\cdot10^{-4}$. $\bar\alpha_T$ sıfıra yeterince yakın değilse $x_T\sim\mathcal N(0,I)$ başlangıcı yanlış olur ve hata (özellikle deterministik DDIM'de) doğrudan örneklere taşınır.
