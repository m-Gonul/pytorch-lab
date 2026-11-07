NumPy; döngüler yalnız log n seviye ya da bit üzerinde.

- `ks_scan(x)` — `x[s:] = x[s:] + x[:-s]` (sağ taraf önce hesaplanır: örtük çift tampon).
- `bk_scan(x)` — n 2'nin kuvveti; her seviyede gelişmiş indeksleme.
- `compact(x, keep)` — dışlayıcı tarama ile çıkış konumu ve dağıtım (boolean indeksleme kullanmadan konumları hesapla; atamada `keep` maskesi serbest).
- `radix_sort(keys, bits=16)` — LSD: her bitte 0'ların hedefi = 0 bayraklarının dışlayıcı taraması, 1'lerinki = (0 sayısı) + 1 bayraklarının dışlayıcı taraması.
- `linear_recurrence(a, b)` — $x_t=a_tx_{t-1}+b_t$, $x_{-1}=0$: (A, B) çiftleri üzerinde Kogge–Stone; bileşke $(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\;a_2b_1+b_2)$.
