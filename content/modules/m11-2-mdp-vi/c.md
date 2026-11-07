NumPy, `P (A, S, S)` satır-stokastik, `R (S, A)`. Yalnız yineleme döngüsü serbest.

- `q_values(P, R, V, gamma)` — `(S, A)`, `einsum("ast,t->sa", P, V)`.
- `value_iteration(P, R, gamma, eps=1e-8, V0=None)` → `(V, pi, n, last_diff)`: durma kuralı $\lVert V_n-V_{n-1}\rVert_\infty\le\varepsilon$.
- `policy_eval(P, R, pi, gamma)` — $(I-\gamma T_\pi)^{-1}R_\pi$ (`np.linalg.solve`), $T_\pi$ = `P[pi, arange(S)]`.
- `bound(gamma, diff)` — $\frac{\gamma}{1-\gamma}\,$`diff`.

Kontrol, rastgele MDP'lerde (1) sınırın sağlandığını, (2) çıkarılan politikanın kaybının $\frac{2\gamma\varepsilon}{1-\gamma}$'yı aşmadığını sınar.
