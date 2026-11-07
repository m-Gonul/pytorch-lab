Saf Python, Bernoulli kolları `probs`.

- `run(policy, probs, T, seed=0, eps=0.1, c=5.0)` → kümülatif **beklenen** pişmanlık listesi ($\sum_t r^*-p_{a_t}$). `policy`:
  - `"eps"` — sabit ε (ε = 0 açgözlü); eşitlik/ilk adım için rastgele; tahmin 0 ile başlar.
  - `"eps_decay"` — $\varepsilon_t=\min(1,c/t)$.
  - `"ucb"` — ilk K adımda her kolu bir kez çek, sonra $\arg\max\,\bar r_a+\sqrt{2\ln t/n_a}$.
  - `"ts"` — Beta(1 + yazı, 1 + tura)'dan `rng.betavariate` ile birer örnek.
