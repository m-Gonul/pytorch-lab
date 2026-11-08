Ortalanmış veri $X_c\in\mathbb R^{n\times d}$ ve kovaryansı $C=\frac1{n-1}X_c^\top X_c$. Birim bir yön $v$ boyunca izdüşümlerin varyansı $v^\top Cv$'dir. Bunu maksimize eden yön $C$'nin en büyük özdeğerine ait [[özvektör]]dür:

$$
\max_{\lVert v\rVert=1}v^\top Cv\quad\Rightarrow\quad Cv=\lambda v .
$$

İlk $k$ özvektör (bileşenler) veriyi $k$ boyuta en az bilgi kaybıyla indirir: $Z=X_cV_k$, yeniden kurma $\hat X=ZV_k^\top+\bar x$.

Hesaplamanın iki yolu:

- **Power iteration**: $v\leftarrow Cv/\lVert Cv\rVert$ tekrarı en büyük özvektöre yakınsar (hız $\lambda_2/\lambda_1$ oranına bağlı). Sonraki bileşen için **deflation**: $C\leftarrow C-\lambda vv^\top$.
- **SVD**: $X_c=USV^\top$ ise $V$'nin sütunları bileşenler, $\lambda_i=s_i^2/(n-1)$.

İşaret kuralı (her aşamada): her bileşende mutlak değerce en büyük eleman pozitif olacak şekilde gerekirse $-1$ ile çarp.

:::türetim Neden kovaryansın özvektörleri? (en büyük varyans yönü)
**İddia:** merkezlenmiş veri $X$ (n × d), kovaryans $S=\frac1nX^\top X$. Birim vektör $u$ boyunca izdüşümün varyansını en büyükleyen $u$, $S$'nin en büyük özdeğerine ait özvektördür ve ulaşılan varyans o özdeğerdir.

1. İzdüşümler $Xu$; ortalamaları 0 (veri merkezli), varyansları $\frac1n\|Xu\|^2=u^\top Su$.
2. Problem: $\max_u u^\top Su$, kısıt $u^\top u=1$ (kısıt olmasa $u$'yu büyüterek sonsuz varyans).
3. Lagrange: $\mathcal L=u^\top Su-\lambda(u^\top u-1)$, $\nabla_u\mathcal L=2Su-2\lambda u=0\Rightarrow Su=\lambda u$: durağan noktalar tam olarak özvektörlerdir.
4. Bir özvektörde amaç $u^\top Su=\lambda u^\top u=\lambda$: en büyüğü en büyük özdeğer.

**İkinci bileşen:** birinciye dik olma kısıtı eklenince aynı akıl yürütme ikinci en büyük özdeğerin özvektörünü verir; $S$ simetrik olduğundan özvektörler zaten diktir. **Eşdeğer görüş:** aynı alt uzay, $k$ boyuta izdüşürmenin karesel yeniden yapılandırma hatasını en küçükler (toplam varyans = korunan + kaybedilen). **Varsayımlar ve sınırlar:** yalnız doğrusal yapı ve ikinci moment; ölçeğe duyarlı (öznitelikleri standartlaştırmadan birimi büyük olan baskın çıkar); varyans her zaman "bilgi" değildir (sınıf ayrımı küçük varyanslı bir yönde olabilir — LDA bu yüzden farklıdır).
:::
