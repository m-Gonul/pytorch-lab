**GAE: döngü mü matris mi?** PyTorch, `r, v, d (T, N)`, `v_last (N,)`.

- `gae_scan(r, v, d, v_last, gamma, lam)` — zamanda geriye tarama, ortamlar paralel (T adımlık döngü).
- `gae_matrix(r, v, d, v_last, gamma, lam)` — döngüsüz: $A=M\delta$, $M_{t,k}=(\gamma\lambda)^{k-t}$ ($k\ge t$ ve $[t,k)$ aralığında bitiş yoksa), ortam başına `(N, T, T)` ve `einsum`. Bitişler: `d`'nin dışlayıcı kümülatif toplamı aynı olmalı.
- `walk_batch(N, T, g)` — 19 durumlu rastgele yürüyüşten N bölüm (merkezden, ±1, uçlarda ±1 ödül), bittikten sonra dolgu: `(S (T+1, N), R, D)`.
