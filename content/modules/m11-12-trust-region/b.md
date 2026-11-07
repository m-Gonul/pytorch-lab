Saf Python, 2 boyut. **Rosenbrock** $f=(1-x)^2+100(y-x^2)^2$, başlangıç (−1.2, 1), minimum (1, 1).

- `rosen(p)`, `rosen_grad(p)`, `rosen_hess(p)` (2×2 liste).
- `gd_backtracking(p0, n_iter)` — gradyan inişi, Armijo geri izleme ($t\leftarrow t/2$, $f(p-tg)\le f(p)-10^{-4}t\|g\|^2$, $t_0=1$); nokta listesi.
- `dogleg(g, H, delta)` — H pozitif tanımlıysa dogleg; değilse Cauchy noktası $p=-\tau\,\delta\,g/\|g\|$, $\tau=1$ ($g^\top Hg\le0$) ya da $\min(\|g\|^3/(\delta g^\top Hg),1)$.
- `trust_region(p0, n_iter, delta0=1.0, delta_max=10.0, eta=0.1)` — ρ < 0.25 → δ/4; ρ > 0.75 ve adım sınırda → min(2δ, δ_max); ρ > η → kabul. Nokta listesi.
