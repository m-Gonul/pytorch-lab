NumPy, döngüsüz. Görüntü `(C, H, W)`, tek örnek.

- `group_norm(x, G, eps=1e-5)` — kanal grupları üzerinde normalize (γ = 1, β = 0).
- `conv3x3(x, W, b)` — "same" dolgu, `W (Co, Ci, 3, 3)`.
- `resblock(x, temb, P)` — $h=\mathrm{conv}_1(\mathrm{SiLU}(\mathrm{GN}(x)))+(W_tt_{emb})$ (kanal başına), $h=\mathrm{conv}_2(\mathrm{SiLU}(\mathrm{GN}(h)))$, çıktı $x+h$ (Ci = Co). `P` sözlüğü: `W1, b1, Wt, bt, W2, b2, G`.
- `attn2d(x, P)` — $x+W_o\,\mathrm{softmax}(QK^\top/\sqrt C)V$, Q, K, V GroupNorm'dan sonra 1×1 izdüşümler; `P`: `Wq, Wk, Wv, Wo, G`.
