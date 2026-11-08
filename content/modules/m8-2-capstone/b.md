Saf Python.

- `ema_update(ema, params, decay)` — listeler; yeni EMA listesi.
- `timestep_embedding(t, d)` — `[sin(t ω_0), …, sin(t ω_{d/2−1}), cos(t ω_0), …]`, $\omega_i=10000^{-i/(d/2)}$.
- `ddim_timesteps(T, steps)` — T − 1'den 0'a eşit aralıklı, yuvarlanmış tamsayı adımlar (azalan).
- `nfe(steps, cfg, batched)` — ağ çağrısı sayısı.
- `dice(pred, gt)` — ikili düz listeler; ikisi de boşsa 1.
