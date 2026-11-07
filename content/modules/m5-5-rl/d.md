Ortam (hazır): 8 hücreli zincir, başlangıç 3. hücre, eylemler sol/sağ. Sağ uç +1 ödül ve bitiş, sol uç 0 ödül ve bitiş, en fazla 30 adım. `env.obs()` one-hot durum.

- `Policy(L=8, hidden=32)` — `Linear → Tanh → Linear`, iki logit.
- `discounted_returns(rewards, gamma)` — liste → tensör $G_t$.
- `run_episode(policy, env)` → `(log_probs (T,), rewards listesi)`; `torch.distributions.Categorical(logits=…)`.
- `train_reinforce(policy, env, episodes=300, gamma=0.95, lr=0.01)` — Adam; baseline: başlangıç getirisinin kayan ortalaması `b ← 0.9 b + 0.1 G_0` (ilk bölümde `b = G_0`), kayıp $-\sum_t\log\pi(a_t\mid s_t)(G_t-b)$.

Kontrol eğitilmiş politikanın açgözlü hâliyle 4 adımda sağ uca gittiğini sınar.
