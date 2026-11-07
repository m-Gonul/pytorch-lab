Binlerce zincir ve geçiş matrisi kestirimi (PyTorch).

- `simulate(T, s0, steps, n, seed=0)` — `T (M, M)` sütun-stokastik tensör; `n` zinciri aynı anda: her adımda `torch.multinomial(T[:, s].T, 1)` (zincir başına kendi sütunu). `(n, steps+1)` `long`.
- `count_mle(traj, M)` — geçiş sayıları $N_{ij}$ ($j\to i$), sütun normalize: $\hat T_{ij}=N_{ij}/\sum_iN_{ij}$. Döngüsüz (`index_put_` ya da `bincount`).
- `grad_mle(traj, M, steps=500, lr=0.5)` — $\hat T=\mathrm{softmax}(\Theta, \text{dim}=0)$, log-olabilirlik $\sum\log\hat T_{s_{t+1},s_t}$'yi Adam ile en çoklaştır; sayma kestirimine yakınsamalı (kapalı form ile gradyan yöntemi aynı optimumu bulur).
