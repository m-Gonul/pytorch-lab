Saf Python, skaler $x$; $t$ 0'dan başlar.

- `alpha_bars(betas)` → liste.
- `q_sample(x0, t, eps, abar)`.
- `oracle_eps(xt, t, abar, mu0, s0)` — veri $\mathcal N(\mu_0,s_0^2)$ iken optimal tahmin (yukarıdaki formül).
- `ddpm_step(xt, t, eps_pred, betas, abar, z)` → $x_{t-1}$; `t == 0` iken gürültü yok.
- `sample(eps_fn, betas, n, rng)` — `eps_fn(x, t)`; $x_T\sim\mathcal N(0,1)$ (`rng.gauss(0, 1)`), $t=T-1,\dots,0$.

Kontrol kâhinle 3000 örnek üretip ortalama ve standart sapmanın $\mathcal N(3, 0.5^2)$'ye uyduğunu sınar.
