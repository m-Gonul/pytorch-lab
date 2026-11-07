Saf Python.

- `tlu(w, b, x)` → 0/1.
- `train_perceptron(X, y, eta=1.0, max_epochs=100)` → `(w, b, hata_sayısı, yakınsadı_mı)`: her örnekte kural; bir epoch hatasız geçerse dur. w, b sıfırdan başlar.
- `margin(w, b, X, y)` — geometrik kenar payı $\min_i y_i^\pm(w^\top x_i+b)/\|w\|$, $y^\pm=2y-1$.
- `xor_net(x)` — iki gizli TLU + çıkış TLU, ağırlıkları sen seç.
