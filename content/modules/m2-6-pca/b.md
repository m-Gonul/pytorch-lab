Saf Python. `X` liste listesi.

- `covariance(X)` — `(d × d)`, payda $n-1$.
- `power_iteration(C, iters=200)` → `(λ, v)`; $v_0=(1,1,…,1)/\sqrt d$, normalize, sonunda işaret kuralı, $\lambda=v^\top Cv$.
- `top_k(C, k, iters=200)` → `(lambdas, vectors)`; her bileşenden sonra deflation.

`numpy`/`torch` yasak.
