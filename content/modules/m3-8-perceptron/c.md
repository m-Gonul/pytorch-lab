**Rastgele öznitelik regresyonu ve çift iniş**, NumPy, döngü yalnız p ve deneme üzerinde.

- `relu_features(X, W)` — $\max(0, XW)$.
- `min_norm_fit(F, y)` — en küçük normlu en küçük kareler (`np.linalg.pinv` ya da `lstsq`).
- `double_descent(n=100, d=10, ps=…, trials=20, sigma=0.3, seed=0)` → `{p: ortanca test MSE}`: hedef $\tanh(2x^\top\beta)$ + gürültü, $W\sim N(0,1/d)$.
- `relu_interpolant(knots, values)` — 1B: verilen düğümlerden geçen parçalı doğrusal fonksiyonu **k ReLU** + doğrusal terimle temsil et → `(a, c0, c1)` öyle ki $f(x)=c_0+c_1x+\sum_ja_j\max(0,x-k_j)$.
