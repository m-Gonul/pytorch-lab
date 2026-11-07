Saf Python. `step(prefix)` — bir demet (token dizisi) alıp `{token: olasılık}` sözlüğü döndüren model.

- `greedy_decode(step, sos, eos, max_len)` → `(dizi, log olasılık)`; dizi `sos` içermez, `eos` dahil.
- `beam_search(step, sos, eos, k, max_len, alpha=0.0)` → `[(dizi, logp)]` skora ($\log p/L^\alpha$, L = token sayısı) göre azalan: her adımda açık hipotezleri uzat, en iyi k adayı tut; EOS ile bitenleri tamamlanmışlara taşı; k tamamlanmış ya da `max_len` olunca dur; sonuç = tamamlanmışlar (yoksa açık olanlar).
- `pad_batch(seqs, pad=0)` → `(dolgulu, maske)`.
- `shift_right(y, sos)` — öğretmen zorlaması girdisi: `[sos] + y[:-1]`.
