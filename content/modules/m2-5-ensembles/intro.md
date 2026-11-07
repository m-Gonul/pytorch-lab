Tek bir ağaç yüksek varyanslıdır: veriyi biraz değiştir, bambaşka bir ağaç çıkar. İki zıt strateji:

**Bagging / random forest** varyansı düşürür. $B$ bootstrap örneklemi üzerinde $B$ derin ağaç eğitip tahminlerini ortalar. Her birinin varyansı $\sigma^2$, aralarındaki korelasyon $\rho$ ise

$$
\operatorname{Var}\Big[\tfrac1B\textstyle\sum_b f_b\Big]=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2 .
$$

**Boosting** bias'ı düşürür. Sığ ağaçları **sırayla** ekler, her biri bir öncekilerin hatasını düzeltir. Gradient boosting bunu fonksiyon uzayında gradient descent olarak yazar:

$$
F_0=\arg\min_c\sum_i L(y_i,c),\qquad
r_i^{(m)}=-\frac{\partial L(y_i,F)}{\partial F}\Big|_{F=F_{m-1}(x_i)},\qquad
F_m=F_{m-1}+\eta\,h_m,
$$

$h_m$, [[pseudo-residual]]'lara oturtulan bir regresyon ağacıdır. MSE'de $r=y-F$ (gerçek residual), logistic kayıpta $r=y-\sigma(F)$.

Regresyon stump'ı için SSE'yi en çok azaltan bölünme, sol/sağ residual toplamları $S_L, S_R$ ile $\frac{S_L^2}{n_L}+\frac{S_R^2}{n_R}$'yi maksimize edendir; M2.4'teki cumsum hilesi burada da çalışır.
