Model: $\hat y = Xw + b$. Kayıp: [[mse|MSE]]

$$
L(w,b)=\frac1n\sum_{i=1}^{n}\big(x_i^\top w+b-y_i\big)^2=\frac1n\lVert Xw+b-y\rVert^2 .
$$

Hata vektörü $e=Xw+b-y$ için zincir kuralı (M0.3) doğrudan verir:

$$
\nabla_w L=\frac2n X^\top e,\qquad \frac{\partial L}{\partial b}=\frac2n\sum_i e_i .
$$

İki yol var. **Kapalı form**: gradyanı sıfıra eşitleyip $(\tilde X^\top\tilde X)\theta=\tilde X^\top y$ normal denklemini çözmek ($\tilde X$, sonuna 1 sütunu eklenmiş $X$). **[[gradient descent|Gradient descent]]**: $\theta\leftarrow\theta-\eta\nabla L$ adımlarını tekrarlamak. Kapalı form tek seferde ama $d$ büyüyünce $O(d^3)$; GD ise sinir ağlarına genelleşen yol.
