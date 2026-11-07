1000 bağımsız deney, aynı anda (PyTorch).

- `batched_td(n_runs, n_steps, alpha, n=5, seed=0)` — `n` durumlu rastgele yürüyüş (1..n, uçlar terminal, sağ uç ödül 1); her deney kendi konumunda; her adımda tüm deneyler bir adım atar, TD(0) güncellemesi `V[run, s]` üzerinde (`scatter_add_` ya da gelişmiş indeksleme — bir deneyde bir adımda tek durum güncellendiği için çakışma yok); terminale ulaşan deney merkeze sıfırlanır. `(V (n_runs, n+2), rms (n_steps,))` — her adımdan sonra deneyler arası ortalama RMS.

Adım döngüsü serbest; deney döngüsü yok.
