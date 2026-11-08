NumPy. Veri: yarıçapı 4 olan halkada 8 Gauss (σ = 0.5); bileşen k'nin sınıfı k mod 4 (her sınıfın karşılıklı iki modu var). Gauss karışımı gürültülendiğinde de karışım kalır: $x_t\sim\sum\pi_kN(\sqrt{\bar\alpha}\mu_k,(\bar\alpha\sigma^2+1-\bar\alpha)I)$ → skor kapalı formda.

- `mixture_eps(x, abar, w)` — `x (N, 2)`, bileşen ağırlıkları `w (8,)`: kesin $\varepsilon(x_t)=-\sqrt{1-\bar\alpha}\,\nabla\log p(x_t)$ (sorumluluklarla). Döngüsüz.
- `class_logprob_grad(x, abar, y)` — $\nabla_{x}\log p(y|x_t)$, kesin (döngüsüz).
- `ddim(xT, abars, eps_fn)` — deterministik DDIM, `abars` artan t sırasında; son adımda $\hat x_0$ döndür (tek döngü).
