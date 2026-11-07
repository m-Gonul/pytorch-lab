Wiener süreci, vektörel.

- `wiener(n_paths, n_steps, t_max=1.0, seed=0)` — `(n_paths, n_steps+1)` `float64`, $W_0=0$, artışlar $\mathcal N(0,\Delta t)$, `cumsum`.
- `quadratic_variation(W)` — her yol için $\sum(\Delta W)^2$ (→ $t_{max}$).
- `max_exceed(W, a)` — $P(\max_{s\le t}W_s\ge a)$ kestirimi (`cummax` ya da `max`) ve $2P(W_t\ge a)$ kestirimi: yansıma ilkesi ikisini eşitler (ayrık adımda max hafif küçük kalır).
