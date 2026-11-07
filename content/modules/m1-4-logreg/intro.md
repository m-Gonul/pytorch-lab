Lineer modelin çıktısı $z=w^\top x+b$ herhangi bir gerçek sayı; ona [[logit]] denir. Sınıflandırmada bir olasılık istiyoruz, bu yüzden $z$'yi [[sigmoid]]'den geçiriyoruz:

$$
p=\sigma(z)=\frac{1}{1+e^{-z}},\qquad \ell(p,y)=-\big[y\ln p+(1-y)\ln(1-p)\big].
$$

$\ell$, ikili [[cross-entropy]]: doğru sınıfa verilen olasılığın negatif logaritması. Zincir kuralı sürpriz derecede sade bir gradyan verir:

$$
\frac{\partial \ell}{\partial z}=p-y,\qquad \nabla_w L=\frac1n X^\top(p-y),\qquad \frac{\partial L}{\partial b}=\frac1n\sum_i(p_i-y_i).
$$

Bu, linear regression'daki $\frac2n X^\top e$ ile aynı biçimde: "tahmin eksi hedef, girdilerle ağırlıklandırılmış". Fark yalnız $e$'nin artık bir olasılık farkı olması.

Sayısal tuzak: $z=-1000$ için `exp(-z)` taşar. Kararlı yazım $z$'nin işaretine göre iki eşdeğer formdan birini seçer; bu modülde `if`'in gerçek bir işi var.
