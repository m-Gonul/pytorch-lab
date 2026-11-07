$K$ sınıf için model her sınıfa bir [[logit]] üretir: $z=W^\top x+b\in\mathbb R^K$. [[softmax|Softmax]] bunları bir olasılık dağılımına çevirir:

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\qquad
\ell(z,y)=-\ln p_y=\underbrace{\ln\sum_j e^{z_j}}_{\text{LSE}(z)}-z_y .
$$

Kategorik [[cross-entropy]]'nin gradyanı M1.4'teki gibi sade:

$$
\frac{\partial\ell}{\partial z}=p-e_y,\qquad
\nabla_W L=\frac1n X^\top(P-Y),\qquad \nabla_b L=\frac1n\mathbf 1^\top(P-Y),
$$

burada $P$ ve $Y$ $(n\times K)$ olasılık ve one-hot matrisleri.

Sayısal tuzak: $e^{1000}$ float64'te taşar. [[log-sum-exp]] hilesi en büyük logit'i dışarı çeker: $\text{LSE}(z)=m+\ln\sum_j e^{z_j-m}$, $m=\max z$. Artık en büyük üs $e^0=1$.
