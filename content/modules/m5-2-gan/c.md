NumPy, **döngü yok**:

- `value_expect(p, q, D)` — ayrık dağılımlar (aynı kutular): $\sum_i p_i\log D_i+\sum_i q_i\log(1-D_i)$.
- `optimal_d(p, q)` — $p/(p+q)$; $p+q=0$ olan kutularda 0.5 (değer etkisiz).
- `jsd(p, q)` — $\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$, $m=\tfrac12(p+q)$; $0\log0=0$.
- `gen_logit_grads(l)` — sahte örneklerin ayırıcı logit'leri $l$ için üretecin kaybının logit'e göre türevleri: `(doymuş, doymamış)`. Doymuş kayıp $\log(1-\sigma(l))$, doymamış $-\log\sigma(l)$.

Kontrol, $V(D^*)=-\log4+2\,\mathrm{JSD}$ eşitliğini ve $D^*$'ın gerçekten maksimum olduğunu (her bozulma değeri düşürür) doğrular.
