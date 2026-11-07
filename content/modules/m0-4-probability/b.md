Listelerle yaz; `numpy`/`torch`/`statistics` yasak, `math` serbest.

- `mean(xs)`, `var(xs, ddof=0)` — `ddof=1` yansız varyansı versin.
- `bernoulli_ll(ys, p)` — log-likelihood $\sum[y\ln p+(1-y)\ln(1-p)]$. $p\in\{0,1\}$ ve verinin buna aykırı olduğu durumda `-math.inf` döndür (`if` gerekiyor; `math.log(0)` hata verir).
- `categorical_mle(rolls, K)` — 1..K arası değerler için `K` uzunlukta olasılık listesi.

Kontrol ayrıca 2000 atışlık veride çarpımın 0'a düştüğünü ama log toplamının sonlu kaldığını gösterir.
