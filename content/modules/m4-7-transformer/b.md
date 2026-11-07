Saf Python:

- `block_params(d)` ve `gpt_params(V, T, d, n_layer)` — yukarıdaki yerleşim (tied çıkış, son LN dahil).
- `probs(logits, temperature=1.0, top_k=None)` → olasılık listesi. top-k'da tutulan token'lar en büyük $k$ logit (eşitlik yok kabul et); diğerleri 0.
- `sample(logits, u, temperature=1.0, top_k=None)` — `u ∈ [0, 1)` düzgün sayı; **ters CDF**: indeks sırasıyla birikimli olasılığı topla, `u < cdf` olan ilk indeksi döndür. `temperature == 0` ise argmax.
- `generate(next_logits, prompt, n, temperature, top_k, rng)` — `next_logits(tokens) → logits`; her adımda `rng.random()` ile örnekle, listeye ekle, `prompt + n` token döndür.
