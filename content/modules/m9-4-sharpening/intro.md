**Türevlerin ayrık hâli** (bir boyutta):

$$
\frac{\partial f}{\partial x}\approx f(x+1)-f(x),\qquad \frac{\partial^2 f}{\partial x^2}\approx f(x+1)+f(x-1)-2f(x).
$$

| yapı | birinci türev | ikinci türev |
|---|---|---|
| sabit bölge | 0 | 0 |
| rampa | sabit, sıfırdan farklı (kalın kenar) | rampa içinde 0, başında/sonunda zıt işaretli |
| basamak | tek tepe | çift tepe + **sıfır geçişi** (kenarın yeri) |
| ince çizgi, nokta | zayıf | çok güçlü |

**Laplasyen:** $\nabla^2f=f_{xx}+f_{yy}$; çekirdekleri $\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$ (4-komşulu) ve köşegenleri de içeren $\begin{bmatrix}1&1&1\\1&-8&1\\1&1&1\end{bmatrix}$. **Keskinleştirme:** $g=f+c\,\nabla^2f$; merkez negatifse $c=-1$.

**[[unsharp masking]]:** bulanık kopyayı çıkararak maske $g_{mask}=f-\bar f$, sonra $g=f+k\,g_{mask}$. $k=1$ unsharp masking, $k>1$ highboost.

**Gradyan:** $\nabla f=(g_x,g_y)$, büyüklük $M=\sqrt{g_x^2+g_y^2}\approx|g_x|+|g_y|$, yön $\alpha=\operatorname{atan2}(g_y,g_x)$. Operatörler:

- Roberts (2×2, çapraz): $\begin{bmatrix}-1&0\\0&1\end{bmatrix}$, $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$
- Prewitt: $[-1,0,1]$ türev × $[1,1,1]$ ortalama
- Sobel: $[-1,0,1]$ türev × $[1,2,1]$ yumuşatma

OpenCV'nin `Sobel` ve `Laplacian` fonksiyonları **korelasyon** uygular ve kenarda `BORDER_REFLECT_101` kullanır.
