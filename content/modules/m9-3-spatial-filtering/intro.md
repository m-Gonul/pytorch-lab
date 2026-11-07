**Doğrusal uzamsal filtreleme:** $m\times n$ ($m=2a+1$, $n=2b+1$) bir $w$ çekirdeği her pikselin komşuluğunda kaydırılır.

$$
\text{korelasyon: }(w\star f)(x,y)=\sum_{s=-a}^{a}\sum_{t=-b}^{b}w(s,t)\,f(x+s,y+t),\qquad
\text{konvolüsyon: }(w*f)(x,y)=\sum_{s,t}w(s,t)\,f(x-s,y-t).
$$

Konvolüsyon çekirdeği 180° çevirir; birim dürtüye uygulandığında çekirdeği aynen kopyalar (dürtü yanıtı). Simetrik çekirdeklerde ikisi aynıdır. OpenCV'nin `filter2D`'si ve PyTorch'un `conv2d`'si **korelasyondur**.

**Kenar doldurma:** sıfır (`BORDER_CONSTANT`), kopyalama (`REPLICATE`: aaa|abc), yansıtma (`REFLECT`: cba|abc), kenarı tekrarlamadan yansıtma (`REFLECT_101`: cb|abc, OpenCV'nin çoğu filtrede **varsayılanı**), sarmalama (`WRAP`).

**[[ayrılabilir filtre]]:** $w=\mathbf c\,\mathbf r^\top$ (rank 1) ise 2-boyutlu filtre, önce satırlara $\mathbf r$, sonra sütunlara $\mathbf c$ uygulamakla aynıdır: piksel başına $mn$ yerine $m+n$ çarpma. Kutu ve Gauss ayrılabilir; Laplasyen değil.

**[[integral görüntü]]** (summed-area table): $A(x,y)=\sum_{r\le x,\,c\le y}I(r,c)$. Herhangi bir dikdörtgenin toplamı dört aramayla:

$$
\sum_{r=r_0}^{r_1}\sum_{c=c_0}^{c_1}I=A(r_1,c_1)-A(r_0-1,c_1)-A(r_1,c_0-1)+A(r_0-1,c_0-1).
$$

`cv2.integral` başına bir sıfır satır/sütun ekler: çıktı $(H+1)\times(W+1)$, $-1$ indeksleri gerekmez.

**Alçak geçiren filtreler:** kutu (ortalama) ve Gauss $G(s,t)=K e^{-(s^2+t^2)/2\sigma^2}$. OpenCV `ksize` verip $\sigma=0$ derseniz $\sigma=0.3((k-1)/2-1)+0.8$ kullanır.
