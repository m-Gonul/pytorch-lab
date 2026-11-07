Saf Python, ikili lineer SVM. `y` elemanları $\pm1$.

- `hinge_grad(X, y, w, b, lam)` → `(L, gw, gb)`; $L=\frac1n\sum\max(0,1-y_if_i)+\lambda\lVert w\rVert^2$. Alt-gradyan: aktif ($y_if_i<1$) örnekler $-y_ix_i/n$ katkı yapar, ceza $2\lambda w$.
- `fit_svm(X, y, lam=0.01, lr=0.1, steps=500)` → `(w, b)`; tam batch alt-gradyan inişi.
- `multiclass_hinge(scores, y, delta=1.0)` — tek örnek, CS231n kaybı.
