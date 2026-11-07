CS231n `svm_loss_vectorized`: $W\in\mathbb R^{d\times C}$, $X\in\mathbb R^{n\times d}$, $S=XW$.

$$
M=\max\big(0,\;S-s_y\mathbf 1^\top+\Delta\big),\quad M_{i,y_i}=0,\qquad L=\frac1n\sum M+\text{reg}\lVert W\rVert^2 .
$$

Gradyan için ihlal göstergesi $B=[M>0]$; doğru sınıf sütununa satırdaki ihlal sayısının negatifi yazılır, $dW=X^\top B/n+2\,\text{reg}\,W$.

- `svm_loss(W, X, y, reg=0.0, delta=1.0)` → `(loss, dW)`. **Döngü yok.** Kontrol, naive (çift döngülü) referansla ve sayısal gradyanla karşılaştırır.
