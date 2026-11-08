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

:::türetim Marj neden $2/\|w\|$?
Karar sınırı $w^\top x+b=0$. Bir $x_0$ noktasının sınıra dik uzaklığı $\frac{|w^\top x_0+b|}{\|w\|}$: $x_0$'dan $w$ yönünde (sınırın normali) $t$ kadar gidip sınıra ulaşmak için $w^\top(x_0-t\frac{w}{\|w\|})+b=0\Rightarrow t=\frac{w^\top x_0+b}{\|w\|}$.

Ölçek serbestliği: $(w,b)$'yi $(cw,cb)$ yapmak sınırı değiştirmez. Bu serbestliği en yakın noktalarda $y_i(w^\top x_i+b)=1$ olacak şekilde sabitleriz (kanonik biçim). O zaman en yakın noktanın uzaklığı $1/\|w\|$, iki sınıf arasındaki şerit $2/\|w\|$.

Marjı büyütmek = $\|w\|$'yi küçültmek: $\min\frac12\|w\|^2$, kısıt $y_i(w^\top x_i+b)\ge1$ (sert marj). Ayrılamayan veride gevşek değişkenler $\xi_i\ge0$ eklenir ve ceza $C\sum\xi_i$; optimumda $\xi_i=\max(0,1-y_if(x_i))$ — **hinge kaybı**. Böylece yumuşak marjlı SVM $\frac12\|w\|^2+C\sum_i\max(0,1-y_if(x_i))$ olur: L2 düzenlileştirilmiş hinge regresyonu ($\lambda=1/C$). Destek vektörleri, kısıtı aktif olan (marjda ya da içinde) noktalardır; diğerlerini silmek çözümü değiştirmez.
:::
