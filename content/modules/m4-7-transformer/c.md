NumPy ile küçük bir GPT'nin ileri geçişi. Parametreler sözlükte (`P`), bloklar `P["blocks"]` listesinde; her blok: `ln1_g, ln1_b, attn_w (d,3d), attn_b, proj_w (d,d), proj_b, ln2_g, ln2_b, fc_w (d,4d), fc_b, fc2_w (4d,d), fc2_b`. Çarpım `x @ W`.

- `layer_norm(x, g, b, eps=1e-5)` — son eksende.
- `gelu(x)` — tanh yaklaşımı: $0.5x\big(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3))\big)$.
- `gpt_forward(idx, P, n_head)` — `idx: (B, T)` → `(B, T, V)` logit. **Yalnız bloklar üzerinde tek döngü**; head'ler reshape ile.
