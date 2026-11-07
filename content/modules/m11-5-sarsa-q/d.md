PyTorch, `float64`. `P` `(A, S, S)` satır-stokastik, `R` `(S, A)`.

- `q_value_iteration(P, R, gamma, tol=1e-10)` — $Q\leftarrow R+\gamma P\max_aQ$ (`einsum`).
- `sync_q_learning(P, R, gamma, n_iter, rate="rescaled", seed=0)` — **eşzamanlı** Q-öğrenme (üretici model): her yinelemede **tüm** (s,a) için $s'\sim P(\cdot|s,a)$ örnekle (`torch.multinomial`) ve hepsini birlikte güncelle. `rate`: `"1/n"` → $\alpha_n=1/n$; `"rescaled"` → $\alpha_n=1/(1+(1-\gamma)n)$.
- `greedy(Q)` → `(S,)`.

1/n neden yavaş? $\gamma$ yakın 1 iken önyüklemeli hedefin yanlılığı $\gamma$ hızında söner; eşit ağırlıklı ortalama erken yanlış hedefleri unutmaz — hata $\sim n^{-(1-\gamma)}$ (Even-Dar & Mansour).
