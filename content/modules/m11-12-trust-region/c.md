**Doğal gradyan**, NumPy, tek durumlu softmax politikalar (haydut), B problem aynı anda: `theta (B, K)`, ödüller `r (B, K)`.

- `softmax(theta)`; `kl(p, q)` → `(B,)`.
- `fisher(pi)` → `(B, K, K)`: $\mathrm{diag}(\pi)-\pi\pi^\top$.
- `fvp(pi, v)` → $Fv=\pi\odot v-\pi(\pi^\top v)$ (matrissiz).
- `policy_grad(theta, r)` → $\nabla_\theta\,\pi^\top r=\pi\odot(r-\pi^\top r)$.
- `cg(Avp, b, iters=20, tol=1e-12)` — eşlenik gradyan, `Avp` bir fonksiyon (döngü serbest; toplu: her satır ayrı sistem, vektörel adımlar).
- `natural_step(theta, r, delta, damping=1e-3)` — $s=\mathrm{CG}(F+\text{damping}\,I,\,g)$, $\beta=\sqrt{2\delta/s^\top Fs}$, $\theta+\beta s$.
