**Paralel rastgele arama** (Mania ve ark. 2018): $a=\mathbb 1[w^\top x>0]$ doğrusal politikalarını binlercesi birden aynı tensörde değerlendir.

- `rollout_lengths(W, x0, max_len=500)` — `W (P, 4)`, `x0 (P, E, 4)` başlangıç durumları (`float64`); her politika kendi E ortamında; `(P,)` ortalama bölüm uzunluğu. Ölen ortamlar maskelenir; hepsi öldüyse erken çık.
- `random_search(n, n_envs=10, seed=0)` → `(W, L)`: $w\sim U[-1,1]^4$, $x_0\sim U[-0.05,0.05]^4$.

Adım döngüsü serbest; politika/ortam döngüsü yok.
