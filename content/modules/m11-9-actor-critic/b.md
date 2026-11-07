**Kısa koridor** (Sutton Örnek 13.1), saf Python. Durumlar 0 (S), 1, 2, 3 (hedef). Eylem sağ/sol; **durum 1'de eylemler ters çalışır**. Solda duvar. Her adım −1, γ = 1. Tüm durumların özniteliği aynı: $x(s,\text{sağ})=(1,0)$, $x(s,\text{sol})=(0,1)$, $\pi(\text{sağ})=\sigma(\theta_0-\theta_1)$.

- `corridor_step(s, right)` → `(s2, r, done)`.
- `corridor_value(p)` — $J(p)=(2p-4)/(p(1-p))$.
- `reinforce(n_eps, alpha, seed=0, baseline=False, alpha_w=0.0, max_steps=1000)` — θ başlangıcı $\pi(\text{sağ})=0.05$; her bölümden sonra her adım için (γ = 1) $\delta=G_t-w$ (taban çizgisi yoksa $\delta=G_t$), önce $w\leftarrow w+\alpha_w\delta$, sonra $\theta\leftarrow\theta+\alpha\,\delta\,\nabla\log\pi$. Çıktı: `(theta, her bölüm sonrası p listesi)`.
