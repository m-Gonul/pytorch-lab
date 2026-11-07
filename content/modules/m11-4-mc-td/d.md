TD(λ): ileri bakış ve geriye bakış (PyTorch, tek bölüm, `float64`).

- `lambda_returns(states, rewards, V, gamma, lam)` — bölüm `states[0..T]` (son durum terminal, değeri 0), `rewards[0..T-1]`; her $t$ için $G^\lambda_t$ (özyineleme: $G^\lambda_t=r_t+\gamma[(1-\lambda)V(s_{t+1})+\lambda G^\lambda_{t+1}]$, $G^\lambda_{T}=0$).
- `offline_forward(states, rewards, V, alpha, gamma, lam)` — $\Delta V(s_t)\mathrel{+}=\alpha(G^\lambda_t-V(s_t))$, hepsi bölüm sonunda uygulanır.
- `offline_traces(states, rewards, V, alpha, gamma, lam)` — birikimli izler $e\leftarrow\gamma\lambda e+\mathbf 1_{s_t}$, $\Delta V\mathrel{+}=\alpha\delta_te$ (δ'lar **eski** V ile), bölüm sonunda uygulanır.
- `online_traces(...)` — aynı ama V her adımda güncellenir.

Çevrimdışı ileri ve geriye bakış **tam olarak** eşittir (Sutton & Barto 12.1).
