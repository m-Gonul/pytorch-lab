[[flow matching]], tümü batch'li:

- `fm_loss(model, x1, z, t)` — `t: (B,)` ∈ [0, 1]; $x_t=(1-t)z+tx_1$; hedef $x_1-z$; `model(x_t, t)` ile ortalama kare hata. Döngü yok.
- `euler_sample(model, z, steps)` — $t_i=i/\text{steps}$, $x\leftarrow x+\frac1{\text{steps}}\,v(x,t_i)$. Tek döngü (adımlar).

Kontrol kâhin hızla adım sayısının etkisini gösterir ve bir ağı senin kaybınla eğitip halkayı 50 Euler adımında üretir.
