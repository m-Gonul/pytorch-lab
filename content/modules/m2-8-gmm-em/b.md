Saf Python, 1-boyutlu GMM. Parametreler listeler: `pis`, `mus`, `vars` (her biri uzunluk $K$).

- `e_step(xs, pis, mus, vars)` → `R` (n × K liste); log uzayında hesapla, satır başına max çıkararak normalize et.
- `m_step(xs, R, reg=1e-6)` → `(pis, mus, vars)`; varyansa `reg` ekle.
- `log_likelihood(xs, pis, mus, vars)` — log-sum-exp ile.
- `em(xs, pis, mus, vars, iters)` → `(pis, mus, vars, ll_geçmişi)`; `ll_geçmişi[0]` başlangıç parametrelerinin, sonrakiler her iterasyondan sonraki log-likelihood.
