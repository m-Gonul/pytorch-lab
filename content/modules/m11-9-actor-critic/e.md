**DPG ile LQR**, PyTorch, paralel ortamlar. $s'=1.1s+0.5u+0.1\epsilon$, ödül $-(s^2+0.5u^2)$, γ = 0.95. Politika $u=ks$ (deterministik), davranış $u=ks+\sigma\epsilon$.

- `riccati()` → `(P, k*)` — Riccati yinelemesi (verildi).
- `collect(k, n_envs, T, sigma, g)` → `(S, U, R, S2)` düzleştirilmiş; adım döngüsü serbest, ortam döngüsü yok; $s$ her adım $[-5,5]$'e kırpılır.
- `lstdq(S, U, R, S2, k)` — karesel öznitelikler $\phi=(s^2,su,u^2,1)$, LSTD-Q: $w=\big[\Phi^\top(\Phi-\gamma\Phi')\big]^{-1}\Phi^\top r$, $\Phi'=\phi(s',ks')$.
- `dpg(n_iter=40, k0=-0.5, alpha=None, n_envs=256, T=20, sigma=0.5, seed=0)` → `(k, geçmiş)`: $\nabla_kJ\approx\overline{s^2}\,(w_1+2w_2k)$ (çünkü $\partial Q/\partial u=w_1s+2w_2u$, $\partial\mu/\partial k=s$); `alpha=None` ise tam iyileştirme $k\leftarrow-w_1/(2w_2)$ (politika yinelemesi).
