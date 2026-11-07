Aynı zincir, `B` ortam paralel (`env.alive` maskesi bitmiş ortamları dondurur, ödüllerini sıfırlar).

- `discounted_returns_vec(r, gamma)` — `r: (B, T)` → `G: (B, T)`. **Döngü yok:** $\Gamma_{jt}=\gamma^{j-t}[j\ge t]$ üst üçgen matrisiyle `G = r @ Γ`. Bitiş sonrası ödüller 0 olduğu için doğru.
- `rollout(policy, env, T)` → `(logp (B, T), r (B, T), mask (B, T))`; `mask[:, t]` adım atılmadan önceki `env.alive`. Yalnız zaman üzerinde tek döngü.
- `train_batched(policy, env, iters=60, gamma=0.95, lr=0.01)` — her iterasyonda bir rollout; baseline: her zaman adımında geçerli girdilerin $G$ ortalaması; kayıp $-\frac1B\sum(\text{logp}\cdot(G-b)\cdot\text{mask})$.
