DPO'yu tam izlenebilir bir oyuncakta çalıştırmak: sözlük 5, uzunluk 4 → tüm 625 dizi sayılabilir. Politika: ilk token için logit'ler + konum başına geçiş matrisleri (verilir). Gerçek ödül: 0 sayısı − 4 sayısı; tercihler referanstan örneklenen çiftlerde Bradley–Terry ile.

- `dpo_loss(pw, pl, rw, rl, beta)` — torch, ortalama.
- `optimal_logp(lref, r, beta)` — $\log\pi^*=\log\mathrm{softmax}(\log\pi_{ref}+r/\beta)$ (tüm diziler üzerinde).
- `kl(lp, lq)` — $\sum e^{lp}(lp-lq)$.
