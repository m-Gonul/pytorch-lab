### Problem: eğitimde iyi, yeni veride kötü {#problem}

M1.1'deki en küçük kareler eğitim hatasını **en aza indirir**. Ama amacımız eğitim verisini değil, **görülmemiş** veriyi iyi tahmin etmektir. Model çok esnekse (çok öznitelik, yüksek dereceli polinom, az veri) eğitim verisindeki gürültüyü de "öğrenir": eğitim hatası sıfıra iner, test hatası patlar. Buna **aşırı uyum** (overfitting) denir.

Belirtileri doğrusal modellerde tanıdıktır: katsayılar çok büyük ve zıt işaretli olur (birbirini götüren dev değerler), veriyi biraz değiştirince katsayılar tamamen değişir. Kök neden: $X^\top X$'in küçük özdeğerli yönleri. Veri o yönlerde çok az değişir, bu yüzden o yönlerdeki katsayılar gürültüyle belirlenir (M1.1'deki $\mathrm{Cov}(\hat\theta)=\sigma^2(X^\top X)^{-1}$: küçük özdeğer → büyük varyans).

**Düzenlileştirme** (regularization) fikri: kayba, modelin karmaşıklığını cezalandıran bir terim ekle.
$$\min_w\ \underbrace{\frac1{2n}\|y-Xw\|^2}_{\text{veriye uyum}}+\underbrace{\lambda\,\Omega(w)}_{\text{karmaşıklık cezası}}.$$
$\lambda\ge0$ ikisi arasındaki takası ayarlar. $\Omega(w)=\frac12\|w\|_2^2$ **ridge**, $\Omega(w)=\|w\|_1$ **lasso**. (Bu modülde $\frac1{2n}$ ölçeklemesini kullanıyoruz; $\lambda$'nın sayısal değeri ölçeklemeye bağlıdır — kütüphaneleri karşılaştırırken dikkat.)

Önce aşırı uyumu kendin gör: polinom derecesini artır, eğitim ve test hatasının nasıl ayrıştığını izle, sonra ridge cezasını aç.

::lab l1

### Yanlılık–varyans ayrışımı {#yanlilik-varyans}

Test hatasının neden önce düşüp sonra yükseldiğini kesin olarak söyleyebiliriz.

:::türetim Beklenen kare hata = yanlılık² + varyans + gürültü
Veri $y=f(x)+\varepsilon$, $\mathbb E\varepsilon=0$, $\mathrm{Var}\,\varepsilon=\sigma^2$. Rastgele bir eğitim kümesi $\mathcal D$'den öğrenilen tahminci $\hat f_{\mathcal D}$. Sabit bir $x$ noktasında, $\bar f(x)=\mathbb E_{\mathcal D}[\hat f_{\mathcal D}(x)]$ ile:
$$\mathbb E_{\mathcal D,\varepsilon}\big[(y-\hat f_{\mathcal D}(x))^2\big]=\underbrace{\big(f(x)-\bar f(x)\big)^2}_{\text{yanlılık}^2}+\underbrace{\mathbb E_{\mathcal D}\big[(\hat f_{\mathcal D}(x)-\bar f(x))^2\big]}_{\text{varyans}}+\underbrace{\sigma^2}_{\text{indirgenemez}}.$$
Kanıt: $y-\hat f=(f-\bar f)+(\bar f-\hat f)+\varepsilon$; karesini al, beklenen değerini al. Çapraz terimler sıfırdır çünkü $\mathbb E_{\mathcal D}[\bar f-\hat f]=0$ ve $\varepsilon$, $\mathcal D$'den bağımsız ve ortalaması 0.
**Yorum:** esnek model → küçük yanlılık, büyük varyans; katı model → tersi. Düzenlileştirme biraz yanlılık ekleyip varyansı çok azaltarak toplamı düşürebilir. **Koşul:** kare kayıp; diğer kayıplarda ayrışım bu kadar temiz değildir. Modern aşırı parametreli modellerde test hatası parametre sayısıyla ikinci kez düşebilir ("double descent", M3.8) — ayrışım yine doğrudur, ama varyansın parametre sayısıyla tekdüze artması varsayımı bozulur.
:::

### Ridge (L2) {#ridge}

:::türetim Kapalı form
Amaç $J(w)=\frac1{2n}\|y-Xw\|^2+\frac\lambda2\|w\|^2$. Gradyan: $\nabla J=-\frac1nX^\top(y-Xw)+\lambda w$. Sıfıra eşitle:
$$\Big(\frac1nX^\top X+\lambda I\Big)w=\frac1nX^\top y.$$
$\frac1nX^\top X$ pozitif yarı-tanımlıdır (özdeğerleri $\ge0$); $\lambda I$ eklemek her özdeğeri $\lambda$ kadar artırır. $\lambda>0$ için matris **her zaman tersinirdir** — $n<d$ olsa, öznitelikler tam tekrarlı olsa bile tek çözüm vardır. Koşul sayısı $\frac{s_{\max}^2/n+\lambda}{s_{\min}^2/n+\lambda}$'ya iner.
:::

**SVD bakışı: hangi yönler küçülür?** $X=U\,\mathrm{diag}(s)\,V^\top$ (tekil değer ayrışımı). En küçük karelerde $w=\sum_j\frac{u_j^\top y}{s_j}v_j$: küçük $s_j$'ye **bölmek** gürültüyü büyütür. Ridge'de
$$w(\lambda)=\sum_j\frac{s_j}{s_j^2+n\lambda}(u_j^\top y)\,v_j=\sum_j\underbrace{\frac{s_j^2}{s_j^2+n\lambda}}_{\text{küçültme çarpanı}}\cdot\frac{u_j^\top y}{s_j}v_j.$$
Çarpan, büyük $s_j$'li yönlerde (veri çok değişiyor, bilgi çok) ≈ 1; küçük $s_j$'li yönlerde (veri az değişiyor, gürültü baskın) ≈ 0. Ridge **bilginin az olduğu yönleri bastırır**. Çarpanların toplamı $\mathrm{df}(\lambda)=\sum_j\frac{s_j^2}{s_j^2+n\lambda}$ "etkili serbestlik derecesi"dir: $\lambda=0$'da $d$, $\lambda\to\infty$'da 0. Modülün **c** aşamasında tüm $\lambda$'lar için yolu tek SVD'den bu formülle hesaplayacaksın.

::lab l5

**Bayesçi yorum.** M0.4: MAP kestirimi + $\mathcal N(0,\tau^2I)$ önseli = L2 ceza. Ridge "katsayılar muhtemelen küçüktür" inancıdır; $\lambda$ önselin gücüdür ($\lambda\propto\sigma^2/\tau^2$).

**İki pratik ayrıntı:**

1. **Bias cezalandırılmaz.** Sabit terimi küçültmek tahminleri keyfi bir sayıya doğru kaydırmak olurdu; $y$'ye 100 eklemek çözümü değiştirmemeli. Uygulama: $X$ ve $y$'yi ortala, ortalanmış veride ridge çöz, sonra $b=\bar y-\bar x^\top w$.
2. **Öznitelikleri ölçekle.** Ceza tüm katsayılara aynı uygulanır; ama bir özniteliği metreden santimetreye çevirmek katsayısını 100'e böler ve cezadan "kurtarır". Ölçeklenmemiş özniteliklerde ridge keyfidir; önce standartlaştır (M1.1).

   Sayısal örnek: aynı özniteliği metre cinsinden alsaydı katsayısı $w$, santimetre cinsinden alsaydı $w/100$ olurdu — amaç fonksiyonu ve en küçük kareler çözümü değişmez, ama ceza $\lambda w^2$ metreli hâline santimetreli hâlinden **$10^4$ kat fazla** vurur. Yani birim seçimi, hangi katsayıların cezadan fiilen muaf olacağını belirler; verideki hiçbir bilgi buna bağlı değildir. Standartlaştırma bu keyfiyeti kaldırır: her öznitelik birim varyansa çekilir, ceza hepsine aynı ölçekte konuşur.

### Lasso (L1) {#lasso}

$\|w\|_1=\sum_j|w_j|$ sıfırda türevlenebilir değildir: kapalı form yoktur ama çözüm çok ilginç bir özelliğe sahiptir — bazı katsayılar **tam olarak sıfır** olur.

:::türetim Tek koordinatta çözüm: soft-threshold
Tek öznitelik, $\frac1n\|x\|^2=1$ olsun ve $z=\frac1nx^\top y$ (cezasız çözüm). Amaç $\frac12(w-z)^2+\lambda|w|$ (sabitler hariç). $w\ne0$ için türev $w-z+\lambda\,\mathrm{sign}(w)=0$ → $w=z-\lambda\,\mathrm{sign}(w)$. Bu, ancak $|z|>\lambda$ ise tutarlıdır ($w$ ile $z$ aynı işaretli). $|z|\le\lambda$ ise: $w=0$'da alt-gradyan koşulu (M0.3) $0\in w-z+\lambda[-1,1]$, yani $|z|\le\lambda$ — sağlanır. Sonuç:
$$w^*=S(z,\lambda)=\mathrm{sign}(z)\max(|z|-\lambda,\ 0).$$
**Yorum:** lasso her katsayıyı sabit bir $\lambda$ kadar sıfıra doğru **kaydırır**; ridge ise oransal küçültür ($z/(1+\lambda)$). Kaydırma sıfırı geçemez: küçük katsayılar tam sıfırda durur.
:::

<figure class="fig"><svg viewBox="0 0 420 240" role="img" aria-label="Soft-threshold ve ridge küçültmesinin karşılaştırması"><line x1="20" y1="120" x2="400" y2="120" class="mute"/><line x1="210" y1="10" x2="210" y2="230" class="mute"/><polyline points="28.0,237.0 29.8,235.9 31.5,234.8 33.2,233.6 35.0,232.5 36.8,231.4 38.5,230.2 40.2,229.1 42.0,228.0 43.8,226.9 45.5,225.8 47.2,224.6 49.0,223.5 50.8,222.4 52.5,221.2 54.2,220.1 56.0,219.0 57.8,217.9 59.5,216.8 61.2,215.6 63.0,214.5 64.8,213.4 66.5,212.2 68.2,211.1 70.0,210.0 71.8,208.9 73.5,207.8 75.2,206.6 77.0,205.5 78.8,204.4 80.5,203.2 82.2,202.1 84.0,201.0 85.8,199.9 87.5,198.8 89.2,197.6 91.0,196.5 92.8,195.4 94.5,194.2 96.2,193.1 98.0,192.0 99.8,190.9 101.5,189.8 103.2,188.6 105.0,187.5 106.8,186.4 108.5,185.2 110.2,184.1 112.0,183.0 113.8,181.9 115.5,180.8 117.2,179.6 119.0,178.5 120.8,177.4 122.5,176.2 124.2,175.1 126.0,174.0 127.8,172.9 129.5,171.8 131.2,170.6 133.0,169.5 134.8,168.4 136.5,167.2 138.2,166.1 140.0,165.0 141.8,163.9 143.5,162.8 145.2,161.6 147.0,160.5 148.8,159.4 150.5,158.2 152.2,157.1 154.0,156.0 155.8,154.9 157.5,153.8 159.2,152.6 161.0,151.5 162.8,150.4 164.5,149.2 166.2,148.1 168.0,147.0 169.8,145.9 171.5,144.8 173.2,143.6 175.0,142.5 176.8,141.4 178.5,140.2 180.2,139.1 182.0,138.0 183.8,136.9 185.5,135.8 187.2,134.6 189.0,133.5 190.8,132.4 192.5,131.2 194.2,130.1 196.0,129.0 197.8,127.9 199.5,126.8 201.2,125.6 203.0,124.5 204.8,123.4 206.5,122.2 208.2,121.1 210.0,120.0 211.8,118.9 213.5,117.8 215.2,116.6 217.0,115.5 218.8,114.4 220.5,113.2 222.2,112.1 224.0,111.0 225.8,109.9 227.5,108.8 229.2,107.6 231.0,106.5 232.8,105.4 234.5,104.2 236.2,103.1 238.0,102.0 239.8,100.9 241.5,99.8 243.2,98.6 245.0,97.5 246.8,96.4 248.5,95.2 250.2,94.1 252.0,93.0 253.8,91.9 255.5,90.8 257.2,89.6 259.0,88.5 260.8,87.4 262.5,86.2 264.2,85.1 266.0,84.0 267.8,82.9 269.5,81.8 271.2,80.6 273.0,79.5 274.8,78.4 276.5,77.2 278.2,76.1 280.0,75.0 281.8,73.9 283.5,72.8 285.2,71.6 287.0,70.5 288.8,69.4 290.5,68.2 292.2,67.1 294.0,66.0 295.8,64.9 297.5,63.8 299.2,62.6 301.0,61.5 302.8,60.4 304.5,59.2 306.2,58.1 308.0,57.0 309.8,55.9 311.5,54.8 313.2,53.6 315.0,52.5 316.8,51.4 318.5,50.2 320.2,49.1 322.0,48.0 323.8,46.9 325.5,45.8 327.2,44.6 329.0,43.5 330.8,42.4 332.5,41.2 334.2,40.1 336.0,39.0 337.8,37.9 339.5,36.8 341.2,35.6 343.0,34.5 344.8,33.4 346.5,32.2 348.2,31.1 350.0,30.0 351.8,28.9 353.5,27.8 355.2,26.6 357.0,25.5 358.8,24.4 360.5,23.2 362.2,22.1 364.0,21.0 365.8,19.9 367.5,18.8 369.2,17.6 371.0,16.5 372.8,15.4 374.5,14.2 376.2,13.1 378.0,12.0 379.8,10.9 381.5,9.7 383.2,8.6 385.0,7.5 386.8,6.4 388.5,5.3 390.2,4.1 392.0,3.0" class="mute dash" fill="none"/><polyline points="28.0,178.5 29.8,177.9 31.5,177.4 33.2,176.8 35.0,176.2 36.8,175.7 38.5,175.1 40.2,174.6 42.0,174.0 43.8,173.4 45.5,172.9 47.2,172.3 49.0,171.8 50.8,171.2 52.5,170.6 54.2,170.1 56.0,169.5 57.8,168.9 59.5,168.4 61.2,167.8 63.0,167.2 64.8,166.7 66.5,166.1 68.2,165.6 70.0,165.0 71.8,164.4 73.5,163.9 75.2,163.3 77.0,162.8 78.8,162.2 80.5,161.6 82.2,161.1 84.0,160.5 85.8,159.9 87.5,159.4 89.2,158.8 91.0,158.2 92.8,157.7 94.5,157.1 96.2,156.6 98.0,156.0 99.8,155.4 101.5,154.9 103.2,154.3 105.0,153.8 106.8,153.2 108.5,152.6 110.2,152.1 112.0,151.5 113.8,150.9 115.5,150.4 117.2,149.8 119.0,149.2 120.8,148.7 122.5,148.1 124.2,147.6 126.0,147.0 127.8,146.4 129.5,145.9 131.2,145.3 133.0,144.8 134.8,144.2 136.5,143.6 138.2,143.1 140.0,142.5 141.8,141.9 143.5,141.4 145.2,140.8 147.0,140.2 148.8,139.7 150.5,139.1 152.2,138.6 154.0,138.0 155.8,137.4 157.5,136.9 159.2,136.3 161.0,135.8 162.8,135.2 164.5,134.6 166.2,134.1 168.0,133.5 169.8,132.9 171.5,132.4 173.2,131.8 175.0,131.2 176.8,130.7 178.5,130.1 180.2,129.6 182.0,129.0 183.8,128.4 185.5,127.9 187.2,127.3 189.0,126.8 190.8,126.2 192.5,125.6 194.2,125.1 196.0,124.5 197.8,123.9 199.5,123.4 201.2,122.8 203.0,122.2 204.8,121.7 206.5,121.1 208.2,120.6 210.0,120.0 211.8,119.4 213.5,118.9 215.2,118.3 217.0,117.8 218.8,117.2 220.5,116.6 222.2,116.1 224.0,115.5 225.8,114.9 227.5,114.4 229.2,113.8 231.0,113.2 232.8,112.7 234.5,112.1 236.2,111.6 238.0,111.0 239.8,110.4 241.5,109.9 243.2,109.3 245.0,108.8 246.8,108.2 248.5,107.6 250.2,107.1 252.0,106.5 253.8,105.9 255.5,105.4 257.2,104.8 259.0,104.2 260.8,103.7 262.5,103.1 264.2,102.6 266.0,102.0 267.8,101.4 269.5,100.9 271.2,100.3 273.0,99.8 274.8,99.2 276.5,98.6 278.2,98.1 280.0,97.5 281.8,96.9 283.5,96.4 285.2,95.8 287.0,95.2 288.8,94.7 290.5,94.1 292.2,93.6 294.0,93.0 295.8,92.4 297.5,91.9 299.2,91.3 301.0,90.8 302.8,90.2 304.5,89.6 306.2,89.1 308.0,88.5 309.8,87.9 311.5,87.4 313.2,86.8 315.0,86.2 316.8,85.7 318.5,85.1 320.2,84.6 322.0,84.0 323.8,83.4 325.5,82.9 327.2,82.3 329.0,81.8 330.8,81.2 332.5,80.6 334.2,80.1 336.0,79.5 337.8,78.9 339.5,78.4 341.2,77.8 343.0,77.2 344.8,76.7 346.5,76.1 348.2,75.6 350.0,75.0 351.8,74.4 353.5,73.9 355.2,73.3 357.0,72.8 358.8,72.2 360.5,71.6 362.2,71.1 364.0,70.5 365.8,69.9 367.5,69.4 369.2,68.8 371.0,68.2 372.8,67.7 374.5,67.1 376.2,66.6 378.0,66.0 379.8,65.4 381.5,64.9 383.2,64.3 385.0,63.8 386.8,63.2 388.5,62.6 390.2,62.1 392.0,61.5" class="ca w2" fill="none"/><polyline points="28.0,192.0 29.8,190.9 31.5,189.8 33.2,188.6 35.0,187.5 36.8,186.4 38.5,185.2 40.2,184.1 42.0,183.0 43.8,181.9 45.5,180.8 47.2,179.6 49.0,178.5 50.8,177.4 52.5,176.2 54.2,175.1 56.0,174.0 57.8,172.9 59.5,171.8 61.2,170.6 63.0,169.5 64.8,168.4 66.5,167.2 68.2,166.1 70.0,165.0 71.8,163.9 73.5,162.8 75.2,161.6 77.0,160.5 78.8,159.4 80.5,158.2 82.2,157.1 84.0,156.0 85.8,154.9 87.5,153.8 89.2,152.6 91.0,151.5 92.8,150.4 94.5,149.2 96.2,148.1 98.0,147.0 99.8,145.9 101.5,144.8 103.2,143.6 105.0,142.5 106.8,141.4 108.5,140.2 110.2,139.1 112.0,138.0 113.8,136.9 115.5,135.8 117.2,134.6 119.0,133.5 120.8,132.4 122.5,131.2 124.2,130.1 126.0,129.0 127.8,127.9 129.5,126.8 131.2,125.6 133.0,124.5 134.8,123.4 136.5,122.2 138.2,121.1 140.0,120.0 141.8,120.0 143.5,120.0 145.2,120.0 147.0,120.0 148.8,120.0 150.5,120.0 152.2,120.0 154.0,120.0 155.8,120.0 157.5,120.0 159.2,120.0 161.0,120.0 162.8,120.0 164.5,120.0 166.2,120.0 168.0,120.0 169.8,120.0 171.5,120.0 173.2,120.0 175.0,120.0 176.8,120.0 178.5,120.0 180.2,120.0 182.0,120.0 183.8,120.0 185.5,120.0 187.2,120.0 189.0,120.0 190.8,120.0 192.5,120.0 194.2,120.0 196.0,120.0 197.8,120.0 199.5,120.0 201.2,120.0 203.0,120.0 204.8,120.0 206.5,120.0 208.2,120.0 210.0,120.0 211.8,120.0 213.5,120.0 215.2,120.0 217.0,120.0 218.8,120.0 220.5,120.0 222.2,120.0 224.0,120.0 225.8,120.0 227.5,120.0 229.2,120.0 231.0,120.0 232.8,120.0 234.5,120.0 236.2,120.0 238.0,120.0 239.8,120.0 241.5,120.0 243.2,120.0 245.0,120.0 246.8,120.0 248.5,120.0 250.2,120.0 252.0,120.0 253.8,120.0 255.5,120.0 257.2,120.0 259.0,120.0 260.8,120.0 262.5,120.0 264.2,120.0 266.0,120.0 267.8,120.0 269.5,120.0 271.2,120.0 273.0,120.0 274.8,120.0 276.5,120.0 278.2,120.0 280.0,120.0 281.8,118.9 283.5,117.8 285.2,116.6 287.0,115.5 288.8,114.4 290.5,113.2 292.2,112.1 294.0,111.0 295.8,109.9 297.5,108.8 299.2,107.6 301.0,106.5 302.8,105.4 304.5,104.2 306.2,103.1 308.0,102.0 309.8,100.9 311.5,99.8 313.2,98.6 315.0,97.5 316.8,96.4 318.5,95.2 320.2,94.1 322.0,93.0 323.8,91.9 325.5,90.8 327.2,89.6 329.0,88.5 330.8,87.4 332.5,86.2 334.2,85.1 336.0,84.0 337.8,82.9 339.5,81.8 341.2,80.6 343.0,79.5 344.8,78.4 346.5,77.2 348.2,76.1 350.0,75.0 351.8,73.9 353.5,72.8 355.2,71.6 357.0,70.5 358.8,69.4 360.5,68.2 362.2,67.1 364.0,66.0 365.8,64.9 367.5,63.8 369.2,62.6 371.0,61.5 372.8,60.4 374.5,59.2 376.2,58.1 378.0,57.0 379.8,55.9 381.5,54.7 383.2,53.6 385.0,52.5 386.8,51.4 388.5,50.3 390.2,49.1 392.0,48.0" class="cb w3" fill="none"/><line x1="140" y1="115" x2="140" y2="125" class="ink"/><line x1="280" y1="115" x2="280" y2="125" class="ink"/><text x="276" y="111" class="sm">λ</text><text x="130" y="111" class="sm">−λ</text><text x="400" y="138" class="sm" text-anchor="end">z (cezasız çözüm)</text><text x="216" y="20" class="sm">ŵ</text><text x="368" y="10" class="sm" text-anchor="end">cezasız: ŵ = z</text><text x="392" y="78" class="sm ta" text-anchor="end">ridge: z/(1+λ)</text><text x="392" y="44" class="sm tb" text-anchor="end">lasso: S(z, λ)</text><text x="210" y="138" class="sm tb" text-anchor="middle">tam 0</text></svg><figcaption>Şekil 2 — Ortonormal özniteliklerde (XᵀX/n = I) ridge her katsayıyı aynı oranda küçültür, hiçbirini sıfırlamaz; lasso her katsayıyı λ kadar sıfıra doğru kaydırır ve |z| ≤ λ olanları tam sıfır yapar.</figcaption></figure>

::lab l4

**Coordinate descent.** Çok öznitelikte, diğerleri sabitken $w_j$'ye göre problem tam olarak yukarıdaki tek boyutlu problemdir: $z_j=\frac1nx_j^\top r_j$, $r_j$ = $j$. öznitelik hariç artık. Koordinatları sırayla güncellemek **coordinate descent**'tir; her adım ucuzdur ve lasso amacı dışbükey olduğu için global minimuma yakınsar. glmnet ve scikit-learn'ün `Lasso`'su bunu kullanır (modülün **b** aşamasında yazacaksın). $\lambda$'yı büyükten küçüğe tararken bir önceki çözümden başlamak (sıcak başlangıç) yolu çok hızlı hesaplar.

**Optimallik koşulu (KKT).** Lasso çözümünde her $j$ için: $w_j\ne0$ ise $\frac1nx_j^\top(y-Xw)=\lambda\,\mathrm{sign}(w_j)$; $w_j=0$ ise $\big|\frac1nx_j^\top(y-Xw)\big|\le\lambda$. Yani artıkla korelasyonu $\lambda$'yı aşamayan öznitelikler dışarıda kalır. $\lambda\ge\lambda_{\max}=\max_j\big|\frac1nx_j^\top y\big|$ iken **tüm** katsayılar sıfırdır: yolun başlangıç noktası.

**Bayesçi yorum:** Laplace önseli (M0.4); tepesi sivri, sıfıra olasılık kütlesi yığar.

### Geometri: elmasın köşeleri {#geometri}

Cezalı biçim $\min\ \text{kayıp}+\lambda\,\Omega(w)$, kısıtlı biçimle eşdeğerdir: $\min\ \text{kayıp}$, öyle ki $\Omega(w)\le t$ (her $\lambda$'ya bir $t$ karşılık gelir; Lagrange dualitesi). Kısıtlı biçimde çözüm, cezasız optimumdan dışarı doğru büyüyen kayıp elipslerinin **kısıt bölgesine ilk değdiği** noktadır.

- L2 bölgesi bir **küre**dir (2-B'de çember): pürüzsüz, köşesiz. Değme noktası hemen hemen hiçbir zaman bir eksen üzerinde değildir.
- L1 bölgesi bir **çapraz çokyüzlü**dür (2-B'de elmas): köşeleri eksenler üzerindedir ve dışarı doğru sivridir. Elips, çoğu konumda önce bir köşeye değer — orada bazı koordinatlar tam sıfırdır. Boyut arttıkça köşelerin ve düşük boyutlu yüzlerin "yakalama alanı" büyür; seyreklik daha da olasılaşır.

Aşağıda cezasız optimumu ve elipsin şeklini değiştir; çözümün ne zaman köşeye (seyrek), ne zaman kenara düştüğünü gör.

::lab l3

### Düzenlileştirme yolu ve λ seçimi {#yol}

$\lambda$'yı büyükten küçüğe tarayınca her katsayı bir eğri çizer: **düzenlileştirme yolu**. Ridge yolunda tüm katsayılar düzgünce sıfıra yaklaşır; lasso yolunda katsayılar tek tek "girer" ($\lambda$ azaldıkça sıfırdan ayrılır). Aynı grafikte doğrulama hatası tipik bir **U** çizer: büyük $\lambda$'da yanlılık (model çok katı), küçük $\lambda$'da varyans (aşırı uyum).

::lab l2

**Korelasyonlu öznitelikler.** İki öznitelik neredeyse aynıysa ridge ağırlığı aralarında **paylaştırır** (her biri yarısını alır); lasso **birini seçer**, diğerini sıfırlar — ve hangisini seçeceği verideki küçük değişikliklere bağlıdır (kararsız seçim). **Elastic net** $\lambda\big(\alpha\|w\|_1+\frac{1-\alpha}2\|w\|_2^2\big)$ ikisini birleştirir: seyreklik korur, ilişkili grupları birlikte tutar.

:::türetim Elastic net'in tek koordinat çözümü
Tek boyutlu problemi lasso'daki gibi ele al: amaç $\frac12(w-z)^2+\lambda_1\lvert w\rvert+\frac{\lambda_2}2w^2$, burada $\lambda_1=\alpha\lambda$, $\lambda_2=(1-\alpha)\lambda$ ve $\frac1n\lVert x\rVert^2=1$. $w\ne0$ için türev $w-z+\lambda_1\,\mathrm{sign}(w)+\lambda_2w=0$; $w=0$'da alt-gradyan koşulu $\lvert z\rvert\le\lambda_1$ lasso'dakinin aynısı kalır. İkisini birleştir:
$$w^*=\frac{S(z,\ \lambda_1)}{1+\lambda_2}.$$
**Yorum:** lasso'nun kaydırmalı eşiği $\lambda_1$ ile ve ridge'in oransal büzülmesi $1/(1+\lambda_2)$ ile aynı adımda uygulanır. $\alpha=1$ saf lasso, $\alpha=0$ saf ridge. **Gruplama etkisi:** $x_1=x_2$ olsun (özdeş öznitelikler). Amaç $w_1\leftrightarrow w_2$ takasında simetrik; $\lambda_2>0$ amaç fonksiyonunu kesin dışbükey yapıp çözümü tekil kıldığı için tek çözüm simetriyi korumak zorundadır: özdeş öznitelikler ağırlığı **eşit paylaştırır** ($w_1=w_2$). Saf lasso'da bu yön düz bir vadi bırakır: sonsuz çok çözüm, seçim keyfi ve verideki küçük değişikliklere duyarlı. Elastic net bu kararsızlığı düzeltir — seyreklik cezası köşeleri tutar, kuadratik terim ise seçimi eşitliğe kilitler.
:::

**λ'yı nasıl seçeriz: çapraz doğrulama.** Test kümesini $\lambda$ seçmek için kullanmak, test hatasını iyimser yapar (test kümesine aşırı uyum). Bunun yerine:

1. Eğitim verisini $k$ parçaya (fold) böl (tipik $k=5$ ya da 10).
2. Her $\lambda$ için: her parçayı sırayla doğrulama olarak ayır, kalan $k-1$ parçada eğit, ayrılan parçada hatayı ölç; $k$ hatanın ortalamasını al.
3. Ortalama hatası en küçük $\lambda$'yı seç (ya da **tek standart hata kuralı**: en iyinin bir standart hatası içindeki en büyük, yani en basit $\lambda$).
4. Seçilen $\lambda$ ile tüm eğitim verisinde yeniden eğit; test kümesine yalnız **en sonda bir kez** bak.

:::uyarı Veri sızıntısı
Standartlaştırma (ortalama, std), öznitelik seçimi ya da eksik değer doldurma gibi veriden öğrenilen her ön işleme **her fold'un içinde, yalnız o fold'un eğitim kısmıyla** yapılmalıdır. Tüm veride standartlaştırıp sonra fold'lara bölmek, doğrulama bilgisini eğitime sızdırır ve hatayı iyimser gösterir. scikit-learn'de `Pipeline` bu yüzden vardır (M3.9).
:::

::lab l6

### Sinir ağlarında: weight decay {#weight-decay}

Sinir ağlarında L2 ceza **weight decay** (ağırlık sönümü) adıyla kullanılır. Neden bu ad?

:::türetim SGD'de L2 ceza = ağırlıkları her adımda küçültmek
Kayıp $L(w)+\frac\lambda2\|w\|^2$; gradyan $g+\lambda w$. SGD adımı:
$$w\leftarrow w-\eta(g+\lambda w)=\underbrace{(1-\eta\lambda)}_{\text{sönüm}}w-\eta g.$$
Her adımda ağırlıklar önce sabit bir oranla küçülür, sonra gradyan adımı atılır. PyTorch'ta `SGD(..., weight_decay=wd)` tam olarak gradyana `wd * w` ekler: $\lambda=$ `wd` (kayıp $\frac{\lambda}{2}\|w\|^2$ ölçeğinde). Modülün **d** aşamasında bu eşdeğerliği sayısal olarak doğrulayacaksın.
:::

**Adam'da durum farklıdır.** Adam gibi uyarlamalı yöntemler gradyanı koordinat başına ölçekler (M3.4). L2 terimini gradyana eklersen, sönüm de ölçeklenir: büyük gradyanlı ağırlıklar daha **az** sönümlenir — amaçlanan bu değildir. **AdamW** (Loshchilov ve Hutter 2019) sönümü gradyandan **ayırır**: $w\leftarrow w-\eta\,\text{Adam adımı}-\eta\lambda w$. Modern eğitimde varsayılan budur; `torch.optim.AdamW`.

**Neyi sönümle?** Genelde bias'lar ve normalizasyon katmanlarının ölçek/kaydırma parametreleri (LayerNorm, BatchNorm) hariç tutulur; bunlar "karmaşıklık" değil ölçek ve konumu ayarlar. PyTorch'ta parametre grupları ile:

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim <= 1 else decay).append(p)   # bias ve norm parametreleri 1-B
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
```

**Başka düzenlileştiriciler** (M3.7): **erken durdurma** (doğrulama hatası artmaya başlayınca dur — doğrusal modelde GD ile erken durdurma yaklaşık olarak ridge'e eşdeğerdir: küçük özdeğerli yönler henüz öğrenilmemiştir), **dropout**, **veri artırma**, **etiket yumuşatma**. Hepsinin ortak noktası: modelin eğitim verisindeki rastlantısal ayrıntılara uymasını zorlaştırmak.

:::türetim Erken durdurma ≈ ridge
Merkezlenmiş en küçük karelerde GD (M1.2'de izlediğin iterasyon): $w\leftarrow w-\frac\eta nX^\top(Xw-y)$. Bunu c'deki SVD bazında çöz: $j$. tekil yöndeki katsayı her adımda $(1-\eta s_j^2/n)$ ile çarpılır, sabit $(\eta/n)s_j(u_j^\top y)$ eklenir. $k$ adımda:
$$w_j(k)=\Big(1-\big(1-\eta s_j^2/n\big)^{k}\Big)\,\frac{u_j^\top y}{s_j}.$$
Küçük $s_j$'li yönler **çok yavaş** öğrenilir: büyük $s_j$'ler ilk birkaç adımda tamamlanır, küçük olanların çarpanı $\approx k\eta s_j^2/n$ ile lineer büyür. Ridge çarpanıyla karşılaştır:
$$k\eta\,\frac{s_j^2}{n}\ \approx\ \frac{s_j^2}{s_j^2+n\lambda}\qquad\Longleftrightarrow\qquad \lambda\approx\frac1{k\eta}.$$
**Yorum:** erken durdurma, yaklaşık olarak $\lambda\approx 1/(k\eta)$'lık bir ridge'tir. Küçük tekil değerli yönleri "henüz öğrenmediği için" bırakır — ridge'in onları önceden bastırmasının zamana yayılmış hâli. Küçük öğrenme oranıyla uzun eğitim ↔ küçük λ (hafif ceza); hızlı öğrenme + erken durdurma ↔ büyük λ. Fark: ridge tüm yönleri bir anda keskin biçimde kırpıyor, erken durdurma yumuşak bir geçiş yapıyor ve eşdeğerlik $\eta$'ya bağlı olduğundan yaklaşık kalır.
:::

### Özet {#ozet}

| | ridge (L2) | lasso (L1) |
|---|---|---|
| ceza | $\frac\lambda2\|w\|_2^2$ | $\lambda\|w\|_1$ |
| çözüm | kapalı form, her zaman tek | coordinate descent; ilişkili özniteliklerde tek olmayabilir |
| tek koordinatta | $z/(1+\lambda)$ (oransal küçültme) | $S(z,\lambda)$ (kaydırma, tam sıfır) |
| seyreklik | yok | var (öznitelik seçimi) |
| ilişkili öznitelikler | ağırlığı paylaştırır | birini seçer |
| önsel | Gauss | Laplace |
| derin öğrenmede | weight decay (Adam'da AdamW) | seyrek modeller, budama |

**Akılda kalsın:** düzenlileştirme biraz yanlılık karşılığında varyansı azaltır. Ridge az bilgi taşıyan yönleri bastırır; lasso küçük katsayıları sıfırlar. Bias'ı cezalandırma, öznitelikleri ölçekle, $\lambda$'yı çapraz doğrulamayla seç ve ön işlemeyi fold içinde yap.

**Kaynaklar:** James, Witten, Hastie, Tibshirani, *ISLP*, Bölüm 6.2; Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Bölüm 3.4 (SVD ile ridge, etkili serbestlik derecesi, lasso geometrisi); Tibshirani (1996), "Regression shrinkage and selection via the lasso"; Friedman, Hastie, Tibshirani (2010), "Regularization paths for generalized linear models via coordinate descent" (glmnet); Zou ve Hastie (2005), elastic net; Loshchilov ve Hutter (2019), "Decoupled weight decay regularization" (AdamW); Duchi ve ark. (2008), L1 topuna izdüşüm; d2l.ai §3.7.
