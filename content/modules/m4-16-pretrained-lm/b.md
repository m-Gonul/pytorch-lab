Saf Python.

- `mlm_mask(tokens, rng, V, mask_id, n_special, p=0.15)` → `(girdi, etiket)`: id'si `< n_special` olan özel token'lar hiç seçilmez; diğerleri p olasılıkla seçilir; seçilenlerin %80'i `mask_id`, %10'u `rng.randrange(n_special, V)`, %10'u aynı kalır. Etiket seçilenlerde orijinal token, diğerlerinde −100. `rng`: `random.Random`.
- `nsp_pairs(sentences, rng)` → `[(A, B, ardışık_mı)]`: her i < n − 1 için %50 `(s_i, s_{i+1}, True)`, %50 `(s_i, rastgele s_j (j ≠ i + 1), False)`.
- `bert_params(V, d, L, ff, max_pos=512, types=2)` — embedding'ler + embedding LN + L katman + pooler.
- `bt_loss(rw, rl)`, `kl_reward(r, logp, logp_ref, beta)`, `dpo_loss(lp_w, lp_l, ref_w, ref_l, beta)`.
