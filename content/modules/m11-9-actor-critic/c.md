**Teoremi doğrula**, NumPy. Tablo softmax politika $\pi(a|s)\propto e^{\theta_{s,a}}$, `P (A, S, S)`, `R (S, A)`, başlangıç dağılımı `d0`.

- `softmax_policy(theta)` → `(S, A)`.
- `evaluate(P, R, pi, gamma)` → `(V, Q)`: $V=(I-\gamma P_\pi)^{-1}r_\pi$, $Q=R+\gamma\,P V$.
- `J(theta, P, R, d0, gamma)` = $d_0^\top V$.
- `pg_theorem(theta, P, R, d0, gamma)` → `(S, A)`: $d_\gamma=(I-\gamma P_\pi^\top)^{-1}d_0$ (indirimli ziyaret), $\partial J/\partial\theta_{s,b}=d_\gamma(s)\,\pi(b|s)\,(Q(s,b)-V(s))$.
- `finite_diff(theta, P, R, d0, gamma, eps=1e-6)` — merkezi farklar (tek döngü serbest).
