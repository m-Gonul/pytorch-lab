Ortalanmış veri $X_c\in\mathbb R^{n\times d}$ ve kovaryansı $C=\frac1{n-1}X_c^\top X_c$. Birim bir yön $v$ boyunca izdüşümlerin varyansı $v^\top Cv$'dir. Bunu maksimize eden yön $C$'nin en büyük özdeğerine ait [[özvektör]]dür:

$$
\max_{\lVert v\rVert=1}v^\top Cv\quad\Rightarrow\quad Cv=\lambda v .
$$

İlk $k$ özvektör (bileşenler) veriyi $k$ boyuta en az bilgi kaybıyla indirir: $Z=X_cV_k$, yeniden kurma $\hat X=ZV_k^\top+\bar x$.

Hesaplamanın iki yolu:

- **Power iteration**: $v\leftarrow Cv/\lVert Cv\rVert$ tekrarı en büyük özvektöre yakınsar (hız $\lambda_2/\lambda_1$ oranına bağlı). Sonraki bileşen için **deflation**: $C\leftarrow C-\lambda vv^\top$.
- **SVD**: $X_c=USV^\top$ ise $V$'nin sütunları bileşenler, $\lambda_i=s_i^2/(n-1)$.

İşaret kuralı (her aşamada): her bileşende mutlak değerce en büyük eleman pozitif olacak şekilde gerekirse $-1$ ile çarp.
