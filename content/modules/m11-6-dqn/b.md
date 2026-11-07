**1000 durumlu rastgele yürüyüş** (Sutton Örnek 9.1), saf Python. Durumlar 1..1000, başlangıç 500; her adımda 1–100 arası düzgün bir sıçrama, sola ya da sağa eşit olasılıkla; 1'in altına çıkış ödül −1, 1000'in üstüne çıkış +1 (bölüm biter). γ = 1.

**Durum toplama:** 10 grup, her biri 100 durum; $\hat v(s)=w_{g(s)}$, gradyan grup göstergesi.

- `walk_episode(rng)` → `[(s, r, s2), …]` (`s2 = None` terminal).
- `group(s, k=10)` → 0..k−1.
- `gradient_mc(episodes, alpha, k=10)` — $w_g\leftarrow w_g+\alpha(G-\hat v(s))$.
- `semi_gradient_td(episodes, alpha, k=10)` — $w_g\leftarrow w_g+\alpha(r+\hat v(s')-\hat v(s))$.
