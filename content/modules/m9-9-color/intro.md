**Algı:** retinadaki 6–7 milyon koni üç türdür (≈ %65 kırmızıya, %33 yeşile, %2 maviye duyarlı; mavi koniler en hassası). Bu yüzden renkler **R, G, B** ana renklerinin karışımı olarak görülür. CIE (1931) ana renkleri: mavi 435.8 nm, yeşil 546.1 nm, kırmızı 700 nm. Üç sabit ana renk görünür tüm renkleri üretemez: üçgen **gamut** at nalı biçimli renklilik diyagramını kapsayamaz.

**Nitelikler:** *ton* (baskın dalga boyu), *doygunluk* (beyazla seyrelme; pembe = kırmızı + beyaz), *parlaklık*. Ton + doygunluk = **renklilik**. Trikromatik katsayılar $x=\frac{X}{X+Y+Z}$, $y$, $z$, $x+y+z=1$.

**Toplamsal ve çıkarımsal:** ışıkta R + G + B = beyaz; boyada ikincil renkler C, M, Y ana renktir: $[C,M,Y]=1-[R,G,B]$. Gerçek mürekkepler saf olmadığından siyah ayrı basılır: $K=\min(C,M,Y)$, $C'=\frac{C-K}{1-K}$ (K = 1 ise C′ = M′ = Y′ = 0).

**HSI** (Gonzalez; $[0,1]$ RGB):
$$\theta=\cos^{-1}\frac{\frac12[(R-G)+(R-B)]}{\sqrt{(R-G)^2+(R-B)(G-B)}},\quad H=\begin{cases}\theta&B\le G\\360-\theta&B>G\end{cases}$$
$$S=1-\frac{3\min(R,G,B)}{R+G+B},\qquad I=\frac{R+G+B}{3}.$$
Geri dönüş üç bölgede (RG: $0\le H<120$, GB, BR): örn. RG bölgesinde $B=I(1-S)$, $R=I\big[1+\frac{S\cos H}{\cos(60°-H)}\big]$, $G=3I-(R+B)$; diğerlerinde önce $H$'den 120 ya da 240 çıkarılıp roller döndürülür.

**HSV** (OpenCV): $V=\max$, $S=(V-\min)/V$, $H$ altıgen üzerinde parçalı doğrusal; 8 bitte $H\in[0,180)$. **YCbCr:** parlaklık + iki renk farkı (JPEG, video). **L*a*b*:** cihazdan bağımsız, algısal olarak yaklaşık düzgün; yoğunluğu renkten ayırır.

**Tam renkli işleme:** her kanalı ayrı işlemek ile vektörü işlemek **doğrusal** komşuluk işlemlerinde aynıdır (ortalama, Laplacian); median, kenar ve eşitleme gibi doğrusal olmayanlarda değil. **Renk bölütleme:** örnek renklerin ortalaması $a$; $D(z,a)=\sqrt{(z-a)^TC^{-1}(z-a)}\le D_0$ (Mahalanobis; $C=I$ ise küre).

**[[Di Zenzo gradyanı]]:** $g_{xx}=|\partial_x\mathbf c|^2$, $g_{yy}=|\partial_y\mathbf c|^2$, $g_{xy}=\partial_x\mathbf c\cdot\partial_y\mathbf c$; yön $\theta=\frac12\tan^{-1}\frac{2g_{xy}}{g_{xx}-g_{yy}}$, büyüklük $F_\theta=\sqrt{\frac12[(g_{xx}+g_{yy})+(g_{xx}-g_{yy})\cos2\theta+2g_{xy}\sin2\theta]}$.
