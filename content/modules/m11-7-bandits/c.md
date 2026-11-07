**10 kollu test yatağı** (Sutton Bölüm 2), NumPy, tüm deneyler aynı anda; yalnız adım döngüsü.

Her deneyde $q_*(a)\sim N(\text{offset},1)$, ödül $N(q_*(a),1)$.

- `testbed(n_runs, n_steps, eps, Q0=0.0, alpha=None, seed=0)` → `(ortalama ödül (n_steps,), optimal eylem oranı (n_steps,))`; `alpha=None` örnek ortalaması, yoksa sabit α; açgözlü eşitlikler rastgele.
- `gradient_bandit(n_runs, n_steps, alpha, baseline=True, seed=0, offset=4.0)` — tercihler $H$, $\pi=\mathrm{softmax}(H)$; $H\leftarrow H+\alpha(r-\bar r)(\mathbb 1_a-\pi)$, $\bar r$ ortalama ödül (taban çizgisi yoksa 0). Aynı çıktılar.
