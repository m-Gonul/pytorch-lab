**A2C**, CartPole (fizik verildi), 32 paralel ortam.

- `class ActorCritic(nn.Module)` — gövde 4→64→64 (tanh), başlıklar: politika logit'leri (2), değer (1). `forward(x)` → `(logits, v)`.
- `nstep_returns(r, d, v_last, gamma)` — `r, d (T, N)`; $R_t=r_t+\gamma(1-d_t)R_{t+1}$, $R_T=V(s_T)$.
- `train_a2c(updates=1500, n_envs=32, n_steps=8, gamma=0.99, lr=2e-3, ent=0.01, seed=0)` → `(ağ, bölüm uzunlukları)`: her güncellemede n adım topla (Categorical örnekleme), kayıp $-\overline{\log\pi\cdot A}+0.5\,\overline{A^2}-\text{ent}\cdot\overline{H}$, $A=R-V$ (aktörde `detach`), gradyan normu 0.5'te kırpılır. 500 adımda kesme; biten ortam yeniden başlar.
