Saf Python.

- `build_vocab(sentences, min_count=1)` → `{kelime: id}`: sayısı ≥ `min_count` olanlar, azalan sayı, eşitlikte alfabetik sıra.
- `oov_hash(word)` — $h\leftarrow(31h+\text{ord}(ch))\bmod 2^{32}$.
- `encode(tokens, vocab, n_oov)` — bilinmeyen kelime → `len(vocab) + oov_hash(w) % n_oov`.
- `skipgram_pairs(ids, window)` — `(merkez, bağlam)` listesi (merkez sırasıyla, bağlam soldan sağa).
- `cbow_examples(ids, window)` → `(bağlam listesi, merkez)`.
- `sgns_loss_grad(v, u_o, U_neg)` → `(J, dv, du_o, [du_k])`, vektörler liste.
- `noise_dist(counts, power=0.75)`; `discard_prob(f, t=1e-5)` → $\max(0, 1-\sqrt{t/f})$.
