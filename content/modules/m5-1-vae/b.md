Saf Python; `mu`, `logvar` aynı uzunlukta listeler (diagonal Gauss).

- `kl_gauss(mu, logvar)` — kapalı form, boyutlar üzerinde toplam.
- `kl_grad(mu, logvar)` → `(dmu, dlogvar)` listeleri.
- `reparam(mu, logvar, eps)` → $z$.
- `log_normal(z, mu, logvar)` — diagonal Gauss log yoğunluğu.
- `kl_mc(mu, logvar, eps_list)` — Monte Carlo: her $\varepsilon$ için $z$, sonra $\log q(z)-\log p(z)$'nin ortalaması.

Kontrol, çok örnekli MC tahminin kapalı forma yaklaştığını doğrular.
