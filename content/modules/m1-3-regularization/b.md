Saf Python, iç içe listelerle. `numpy`/`torch` yasak.

- `soft_threshold(z, t)`
- `ridge2(X, y, lam)` — **iki** özellik, bias yok: $X^\top X+\lambda I$'yı kur, $2\times2$ ters formülüyle çöz, `[w1, w2]` döndür.
- `lasso_cd(X, y, lam, iters=100)` — herhangi sayıda özellik, bias yok, amaç $\frac1{2n}\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$. Her iterasyonda $j=0..d-1$ sırasıyla tek koordinat güncellemesi (intro'daki formül). `w` sıfırdan başlasın.

Kontrol: büyük λ'da tüm katsayılar **tam** 0, λ = 0'da OLS'ye yakınsama.
