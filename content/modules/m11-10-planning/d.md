**Öğrenilmiş sinir ağı modeliyle MPC** (Nagabandi ve ark. 2018), CartPole (gerçek fizik `cartpole_step` verildi, modelden gizli).

- `random_data(n_eps, g)` → `(X, A, X2)`: rastgele politika, bölüm en fazla 200.
- `class Model(nn.Module)` — 5 → 128 → 128 → 4 (ReLU), **fark** tahmini: $\hat s'=s+\text{net}((s,a)\text{ normalize})\cdot\sigma_\Delta$; normalizasyon istatistikleri tampon (buffer).
- `fit(model, X, A, X2, epochs=100, seed=0)` — normalize Δ üzerinde MSE, Adam 1e-3, batch 256; son kayıp.
- `mpc_action(model, x, g, n_seq=500, H=20)` — **rastgele atış**: 500 rastgele ikili eylem dizisini modelde paralel ilerlet; maliyet $\sum_h[10\cdot\mathbb 1(\text{düştü})+\theta_h^2+0.01x_h^2]$ (düşüş, bir kez düştükten sonra kalıcı); en iyi dizinin ilk eylemi.
- `run_mpc(model, g, max_len=200)` → bölüm uzunluğu.
