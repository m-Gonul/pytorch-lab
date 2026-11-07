**TRPO ve PPO**, CartPole, 16 paralel ortam × 128 adım. Ortak: `PolicyValue` (ayrı politika ve değer MLP'leri, tanh), `rollout(...)` (verildi), `gae(r, v, d, v_last, gamma, lam)`.

- `ppo_loss(logp_new, logp_old, adv, eps)` → $-\overline{\min(r A,\mathrm{clip}(r)A)}$.
- `train_ppo(iters=60, eps=0.2, epochs=4, mb=256, lr=3e-4, seed=0)` — avantaj normalize; kayıp = PPO + 0.5·değer MSE; gradyan normu 0.5.
- `fvp(pi, X, v, damping=0.1)` — ortalama $\mathrm{KL}(\pi_{\text{sabit}}\|\pi_\theta)$'nın Hessian'ı × v (iki kez `autograd.grad`).
- `cg(Avp, b, iters=10)`.
- `trpo_step(pi, X, A, adv, old_logp, max_kl=0.01)` — $g=\nabla L$, $s=\mathrm{CG}(F,g)$, $\beta=\sqrt{2\delta/s^\top Fs}$; çizgi araması $\beta\cdot0.5^k$: KL ≤ 1.5δ ve L iyileşmişse kabul, yoksa eski parametreler. Kabul edildiyse True.
- `train_trpo(iters=60, max_kl=0.01, seed=0)` — politika TRPO ile, değer Adam (1e-3, 4 epoch).
