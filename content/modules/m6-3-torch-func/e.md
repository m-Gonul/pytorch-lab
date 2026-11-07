`dp_sgd_step(model, x, y, max_norm, noise_mult, lr, generator=None)` — bir DP-SGD adımı:

1. Örnek başı gradyanlar (`vmap(grad(…))`).
2. Her örneğin **tüm parametreler üzerinden** gradyan normu $\lVert g_i\rVert$; ölçek $\min(1, C/\lVert g_i\rVert)$.
3. Kırpılmış gradyanları topla, $\mathcal N(0,\sigma^2C^2)$ gürültü ekle (`torch.randn(…, generator=generator)`), $B$'ye böl.
4. `p -= lr * g` (`torch.no_grad()` içinde).

Örnekler üzerinde döngü yok.
