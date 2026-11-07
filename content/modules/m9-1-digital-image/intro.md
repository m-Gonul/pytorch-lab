Kamera sürekli bir parlaklık fonksiyonu $f(x,y)$'yi iki kez ayrıklaştırır:

- **Örnekleme** (uzamsal): bir ızgaradaki noktalarda değer al → $M\times N$ piksel. Çözünürlük düşükse ince ayrıntılar kaybolur ya da **aliasing** olur.
- **Kuantalama** (yoğunluk): her değeri $L=2^k$ seviyeden birine yuvarla. 8 bit (256 seviye) gri görüntülerde standarttır; az bit **yanlış kontur** üretir.

Depolama: $M\cdot N\cdot k$ bit.

**İnterpolasyon** ızgara noktaları arasındaki değeri tahmin eder. $(x, y)$'nin dört komşusu $x_0=\lfloor x\rfloor$, $y_0=\lfloor y\rfloor$, ağırlıklar $a=x-x_0$, $b=y-y_0$:

$$
f(x,y)\approx(1-a)(1-b)f_{00}+a(1-b)f_{01}+(1-a)b\,f_{10}+ab\,f_{11}.
$$

Bikübik interpolasyon her eksende 4 komşuyu Keys çekirdeğiyle ağırlıklar ($a=-0.75$ OpenCV, $-0.5$ birçok başka kütüphane):

$$
W(t)=\begin{cases}(a+2)|t|^3-(a+3)|t|^2+1 & |t|\le1\\ a|t|^3-5a|t|^2+8a|t|-4a & 1<|t|<2\\ 0 & \text{aksi}\end{cases}
$$

**OpenCV'nin piksel merkezi kuralı:** ölçek $s=$ kaynak/hedef boyutu, hedef $x$'in kaynağı $x_s=(x+0.5)s-0.5$, kenarda kırpılır.

**Geometrik dönüşümler** (döndürme, ölçekleme, kaydırma, kesme) bir afin matrisle yazılır. Uygulamada **ters eşleme**: her çıkış pikseli için kaynak koordinatını hesapla ve interpolasyonla oku.

**Komşuluk ve mesafe:** 4-komşular yatay/dikey, 8-komşular çaprazları da içerir; $D_4=|\Delta x|+|\Delta y|$, $D_8=\max(|\Delta x|,|\Delta y|)$, $D_e=\sqrt{\Delta x^2+\Delta y^2}$.

Test görüntüleri: `skimage.data.camera()` (512 × 512) gibi scikit-image ile gelen kamu malı görüntüler; tarayıcıda da yüklenir.
