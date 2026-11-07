- `VAE(d_in, d_lat, hidden)` — `encode(x)` → `(mu, logvar)`; `decode(z)` → $\hat x$; `forward(x)` → `(xhat, mu, logvar)` ([[reparametrization trick]] ile). Kodlayıcı ve çözücü: `Linear → ReLU → Linear`.
- `elbo_loss(x, xhat, mu, logvar, beta=1.0)` — örnek başına $\lVert x-\hat x\rVert^2 + \beta\cdot\mathrm{KL}$, batch ortalaması.
- `sample(model, n)` — prior'dan $z$, decode.

Kontrol modeli 12 boyutlu, 2 gizil faktörlü sentetik veride eğitir: kayıp düşmeli, yeniden yapılandırma veri varyansının küçük bir kesrine inmeli ve KL sıfıra çökmemeli.
