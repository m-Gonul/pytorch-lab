Saf Python.

- `positional_encoding(n, d)` → `n × d` liste.
- `noam_lr(step, d, warmup, factor=1.0)` (step ≥ 1).
- `smooth_targets(t, V, eps)` — $q=(1-\varepsilon)\,\text{onehot}(t)+\varepsilon/V$.
- `smoothed_ce(logp, t, eps)` — `logp` log olasılık listesi.
- `transformer_params(d, h, dff, N, V)` → `{"enc_layer", "dec_layer", "total"}` (paylaşılan embedding, çıktı katmanında bias yok, Post-LN).
- `causal_allowed(T)` — nedensel maskede izinli çift sayısı.
