**DQN**, CartPole (ortam fiziği verildi: `cartpole_step`, toplu tensörler). Ödül her adımda 1, en fazla 500 adım.

- `Replay(cap, dim)` — dairesel tampon (önceden ayrılmış tensörler): `push(s, a, r, s2, d)` (her biri birinci boyutta k örnek), `sample(b, g)` → `(s, a, r, s2, d)`; `len(buf)`.
- `td_target(q_target, q_online, r, s2, d, gamma, double)` — `no_grad`; DQN: $r+\gamma(1-d)\max Q_{\bar w}(s')$, çift DQN: $r+\gamma(1-d)Q_{\bar w}(s',\arg\max Q_w(s'))$.
- `soft_update(target, online, tau)` — $\bar w\leftarrow\tau w+(1-\tau)\bar w$ (τ = 1: kopyala).
- `train_dqn(steps=30000, seed=0, double=True)` → `(q, bölüm uzunlukları)`: `make_net()`, Adam (1e-3), tampon 50 000, ısınma 1000, batch 64, Huber kaybı, γ = 0.99, ε 1 → 0.05 ilk %30'da doğrusal, hedef ağ her 500 adımda kopya. **Kesme (500. adım) terminal değildir:** `d` yalnız düşüşte 1.
