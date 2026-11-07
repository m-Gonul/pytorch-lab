NumPy, döngüsüz.

- `pe(n, d)`.
- `offset_rotation(k, d)` → `(d, d)` matris $R_k$: tüm p için $P_{p+k}=R_kP_p$ (her (sin, cos) çifti için 2×2 döndürme).
- `mha(xq, xkv, Wq, Wk, Wv, Wo, h, allow)` — `xq (B, T, d)`, `xkv (B, S, d)`, ağırlıklar `(d, d)`, `allow (B, T, S)` bool (True = izinli) → `(B, T, d)`.
- `decoder_masks(tgt_lens, src_lens, T, S)` → `(self (B, T, T), cross (B, T, S))`: öz-maske = nedensel ∧ hedef anahtar dolgu değil; cross = kaynak anahtar dolgu değil.
- `ls_ce(logits, t, eps, pad=0)` — dolgu hariç ortalama yumuşatılmış CE.
