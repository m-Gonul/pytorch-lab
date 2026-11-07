Çok özellikli, NumPy.

- `best_reg_stump(X, r)` → `(j, t, sol, sağ)`; tüm özellik ve eşikler **döngüsüz**. Sıralı residual'ların cumsum'u $S_L$, sayaç $n_L$; kazanç $\frac{S_L^2}{n_L}+\frac{S_R^2}{n_R}$. Eşit değer pozisyonları geçersiz. Eşitlikte küçük özellik, sonra küçük eşik. Bölünme yoksa `None`.
- `gbm_np(X, y, rounds, lr)` → model; `gbm_predict_np(model, X)` → `(n,)`. Her ikisinde de yalnız **tur** döngüsü.

Kontrol: $y=\sin(2x_0)+x_1^2$ gibi doğrusal olmayan bir fonksiyonda 200 turda test $R^2\ge0.9$.
