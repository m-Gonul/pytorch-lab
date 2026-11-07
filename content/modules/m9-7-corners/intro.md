**Fikir:** küçük bir pencereyi her yöne kaydır. Düz bölgede hiçbir yönde, kenarda tek yönde, **köşede her yönde** değişim olur.

**Türetme:** ağırlık $w$ ile kare fark toplamı $C(x,y)=\sum_{s,t}w(s,t)\,[f(s+x,t+y)-f(s,t)]^2$. Birinci dereceden Taylor açılımı $f(s+x,t+y)\approx f+xf_x+yf_y$ ile

$$C(x,y)\approx\begin{bmatrix}x&y\end{bmatrix}M\begin{bmatrix}x\\y\end{bmatrix},\qquad M=\sum_{s,t}w(s,t)\begin{bmatrix}f_x^2&f_xf_y\\f_xf_y&f_y^2\end{bmatrix}.$$

$M$ ([[Harris matrisi]]) türevlerin yerel ağırlıklı kovaryansıdır: özvektörleri en büyük yayılım yönleri, özdeğerleri o yönlerdeki yayılım. **İki küçük** özdeğer → düz, **biri büyük** → kenar, **ikisi büyük** → köşe (ne yazık ki tek parlak noktalar da).

**Özdeğer hesaplamadan:** $\det M=\lambda_1\lambda_2$, $\mathrm{tr}M=\lambda_1+\lambda_2$,
$$R=\det M-k\,\mathrm{tr}^2M:$$
köşede büyük pozitif, kenarda büyük negatif, düzde küçük. $0<k<0.25$; küçük $k$ daha hassas (daha çok köşe). Köşe: $R>T$ (ve yerel en büyük).

**Ağırlık:** kutu (hızlı, gürültü azsa) ya da Gauss $w=e^{-(s^2+t^2)/2\sigma^2}$ (yumuşatma önemliyse; ayrıca dönmeye karşı izotropik).

**Shi-Tomasi:** $R=\min(\lambda_1,\lambda_2)$, kapalı formda $\frac{a+c}2-\sqrt{(\frac{a-c}2)^2+b^2}$.

**FAST:** pikselin etrafındaki 16 piksellik çemberde, $I_p+t$'den parlak ya da $I_p-t$'den koyu **ardışık 9** piksel varsa köşe. Türev yok, çok hızlı.
