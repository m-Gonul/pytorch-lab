Grafiği kâğıda çiz: iki girdi düğümü ($x$, $y$), bir toplama düğümü ($a$), bir çarpma düğümü ($f$). $y$'den **iki** ok çıktığına dikkat et.

Geri geçişte $\partial f/\partial f = 1$ ile başla ve her düğümde aşağıdaki tabloyu kullan:

| düğüm | yerel türevler |
|---|---|
| $a = x + y$ | $\partial a/\partial x = 1$, $\partial a/\partial y = 1$ |
| $f = a\cdot y$ | $\partial f/\partial a = y$, $\partial f/\partial y = a$ |

Sonra aynı sayıları sayısal türevle (ileri ve geri birer adım) doğrula. Son olarak iki çıktılı bir fonksiyonun $2\times2$ Jacobian'ını doldur.
