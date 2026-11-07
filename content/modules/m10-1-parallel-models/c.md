NumPy.

- `critical_path(A, dur)` — `A (n, n)` bitişiklik ($A_{ij}=1$: i → j), `dur (n,)`; en erken bitiş zamanları $\text{ef}_j=\text{dur}_j+\max_{i\to j}\text{ef}_i$ ("max-artı" çarpımıyla n kez gevşetme, tek döngü); `(ef, max(ef))`.
- `fit_amdahl(p, T)` — $T(p)=a+b/p$ en küçük kareler → `(seri oran a/(a+b), T1 = a+b)`.
- `speedup_grid(f, p)` — `f (F,)`, `p (P,)` → Amdahl ve Gustafson `(F, P)` tabloları, döngüsüz.
