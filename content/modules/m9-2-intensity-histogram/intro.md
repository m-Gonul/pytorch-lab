**Nokta işlemi:** her çıkış pikseli yalnızca aynı konumdaki giriş pikseline bağlı, $s=T(r)$. 8 bitte $T$ 256 girişlik bir **arama tablosudur** (LUT): `out = lut[img]`.

| dönüşüm | $T(r)$ | etkisi |
|---|---|---|
| negatif | $L-1-r$ | koyu bölgelerdeki beyaz ayrıntıyı öne çıkarır |
| log | $c\log(1+r)$ | geniş dinamik aralığı sıkıştırır (Fourier spektrumu) |
| gamma | $c\,r^\gamma$ | $\gamma<1$ açar, $\gamma>1$ koyulaştırır |
| kontrast germe | parçalı doğrusal | seçilen aralığı genişletir |

**[[histogram eşitleme]].** $n_k$ seviye $k$'deki piksel sayısı, $p_k=n_k/MN$. Sürekli durumda $s=(L-1)\int_0^rp(w)\,dw$ çıkış yoğunluğunu düzgün yapar. Ayrık hâli:

$$
s_k=\operatorname{round}\Big((L-1)\sum_{j=0}^{k}p_j\Big).
$$

OpenCV'nin `equalizeHist`'i en düşük dolu seviyenin birikimli sayısını ($c_{\min}$) çıkarır ve çıkışı tam aralığa yayar:

$$
s_k=\operatorname{round}\Big((L-1)\frac{c_k-c_{\min}}{MN-c_{\min}}\Big).
$$

İki koşul: $T$ monoton artan (yoğunluk sırası tersine dönmez) ve $0\le T(r)\le L-1$ (aralık korunur).

**Histogram eşleme:** girdinin CDF'sini hedef görüntünün CDF'sinin tersiyle birleştir: $s=G^{-1}(T(r))$.
