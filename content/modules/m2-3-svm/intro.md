Logistic regression her örneğin olasılığını iyileştirmeye çalışır; **SVM** ise yalnız sınıflar arasında güvenli bir boşluk (margin) ister. $y\in\{-1,+1\}$ ve $f(x)=w^\top x+b$ için [[hinge loss|hinge kaybı]]:

$$
\ell(x,y)=\max\big(0,\;1-y\,f(x)\big),\qquad
L=\frac1n\sum_i\ell_i+\lambda\lVert w\rVert^2 .
$$

$yf(x)\ge1$ olan örnekler kayba katkı yapmaz. Kayıp $yf(x)=1$'de kırılır; orada türev yok ama bir [[alt-gradyan]] var: aktif örnekler için $\partial\ell/\partial w=-y\,x$, diğerleri için 0.

Çok sınıflı sürüm (Weston–Watkins, CS231n): $s=Wx$ skorlar,

$$
\ell=\sum_{j\ne y}\max(0,\;s_j-s_y+\Delta).
$$

Doğrusal olmayan sınırlar için [[kernel trick]]: $f(x)=\sum_i\alpha_i\,k(x_i,x)+b$. RBF kernel $k(x,x')=e^{-\gamma\lVert x-x'\rVert^2}$ ile model, her eğitim noktasına bir "tümsek" koyar.
