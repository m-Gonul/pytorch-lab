Saf Python, `random.Random(seed)` ile.

- `sample_index(probs, u)` — $u\in[0,1)$ için kümülatif olasılıklarda ilk $c_i>u$ olan $i$ (ters CDF örnekleme).
- `simulate_chain(T, s0, steps, seed=0)` — `T` liste listesi (sütun-stokastik, `T[i][j]` = $j\to i$); her adımda `rng.random()` ile sütundan örnekle. `steps+1` uzunlukta durum listesi.
- `ar1(phi, sigma, n, seed=0)` — $x_0=0$, $x_t=\phi x_{t-1}+w_t$, $w_t$ = `rng.gauss(0, sigma)`.
- `autocov(x, lag)` — örnek otokovaryansı $\frac1n\sum_{t}(x_t-\bar x)(x_{t+lag}-\bar x)$ (payda $n$).
