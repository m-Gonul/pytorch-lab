NumPy; `P: (S, A, S)`, `R: (S, A)` (beklenen ödül).

- `bellman_q(V, P, R, gamma)` → `Q: (S, A)`; `einsum("sat,t->sa", …)`. Döngü yok.
- `policy_eval_exact(P, R, pi, gamma)` — `pi: (S, A)` stokastik politika; $P_\pi$, $R_\pi$ kur, `np.linalg.solve`. Döngü yok.
- `policy_iteration(P, R, gamma)` → `(V, pi_det)`; değerlendir → açgözlü iyileştir → politika değişmeyince dur. Tek döngü.
