Vektörel simülasyon, `n_runs` deney aynı anda (PyTorch).

- `vec_bandit(probs, T, n_runs, method, eps=0.1, seed=0)` → `(T,)` ortalama kümülatif pişmanlık; `method ∈ {"eps", "ucb", "ts"}` (b ile aynı kurallar).
- `lai_robbins(probs)` → $\sum_{a:\Delta_a>0}\Delta_a/\mathrm{KL}(p_a\|p^*)$ (Bernoulli KL).
