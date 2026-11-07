PyTorch. **Ders alıştırması 1:** artan derinlikle fonksiyon yaklaşımı.

- `make_data(fn, n, seed)` — `fn`: `"f1"` skaler $\sin(3\pi x)/(1+x^2)$, $x\sim U[-1,1]$; `"f2"` $e^{-(x_1^2+x_2^2)/0.1}$, $x\sim U[-1,1]^2$; `"f3"` iki çıktılı $\big[e^{-(x_1^2+x_2^2-2x_3)/0.1},\;e^{-(x_1^2+x_2^2+2x_3)/0.1}\big]$, $x\sim U[-0.5,0.5]^3$ (alıştırmanın 2. ve 3. maddeleri).
- `mlp(d_in, d_out, depth, width=64)` — `depth` gizli katman (ReLU).
- `fit(model, X, Y, steps=2000, lr=3e-3, seed=0)` → eğitim sonrası MSE (Adam, tam batch).
- `depth_table(fns, depths, n=2000)` → `{(fn, depth): test MSE}`.
- `penultimate_separability(model, X, y)` — eğitilmiş bir sınıflandırıcının son gizli katman aktivasyonlarına doğrusal sınıflandırıcı (lojistik regresyon, PyTorch) sığdır; doğruluk.
