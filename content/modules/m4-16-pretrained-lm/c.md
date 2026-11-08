NumPy.

- `mlm_mask_batch(X, rng, V, mask_id, n_special, p=0.15)` — döngüsüz; `rng`: `np.random.Generator`.
- `elmo_mix(H, s, gamma)` — `H (L+1, T, d)`, `s (L+1,)` ham ağırlıklar → $\gamma\sum_j\mathrm{softmax}(s)_jH_j$.
- `seq_logprob(logits, y, resp_mask)` — `logits (B, T, V)` (t. konum y_t'yi tahmin eder), yanıt token'larının log olasılık toplamı `(B,)`.
- `dpo_batch(pw, pl, rw, rl, beta)` → `(ortalama kayıp, örtük ödül doğruluğu, ortalama marj)`; örtük ödül $\beta(\log\pi-\log\pi_{ref})$.
- `fit_reward(Fw, Fl, lr=0.5, iters=500)` — doğrusal ödül $r=Fw$; Bradley–Terry kaybında gradyan inişi (tek döngü).
