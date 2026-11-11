### Bağlam: haydutlara durum eklemek {#baglam}

M11.7'deki haydutlarda bütün kullanıcılar aynıydı: bir reklamın tek bir tıklanma oranı vardı. Pek çok uygulamada eylemi seçmeden önce ek bilgi gelir. Kişiselleştirilmiş reklamda ve kullanıcı arayüzlerinde bu bilgi kullanıcının özellikleridir (konum, yaş, cihaz, geçmiş). Eylemlerin de ödülü etkileyen öznitelikleri olabilir: reklamın ya da sayfanın konusu, anahtar kelimeleri. En iyi reklam herkes için aynı değildir; bağlama göre değişir.

**Bağlamlı haydut**, durumları (bağlamları) ve eylem öznitelikleri olan bir çok kollu hayduttur. Biçimsel olarak:

- $\mathcal S$: durumlar. Her durum bir öznitelik vektörüyle tanımlanır, $\mathbf x_s=(x_1^s,\dots,x_k^s)$.
- $\mathcal A$: eylemler. Her eylemin bir öznitelik vektörü vardır, $\mathbf x_a=(x_1^a,\dots,x_l^a)$.
- Ödüller, çoğu zaman $\mathbb R$'de.

Geçiş fonksiyonu yoktur: her adımın durumu öncekilerden **bağımsız** gelir ve ajanın eylemi bir sonraki bağlamı etkilemez. Bu, bağlamlı haydutu tam RL'den ayıran özelliktir. Her adım tek adımlık bir karar problemidir; değer fonksiyonu ya da bootstrapping gerekmez. Amaç, beklenen ödülü en çoklayan bir politika $\pi:\mathbf x_s\to a$ bulmaktır:
$$E[r\mid s,a]=E[r\mid\mathbf x_s,\mathbf x_a].$$

### Ödül fonksiyonunu regresyonla yaklaşmak {#regresyon}

Yaygın yaklaşım, ortalama ödül fonksiyonunu regresyonla öğrenmektir: $\tilde R(s,a)=\tilde R(\mathbf x)$, burada $\mathbf x=(\mathbf x_s,\mathbf x_a)$ birleşik öznitelik vektörüdür.

- **Doğrusal:** $\tilde R_{\mathbf w}(\mathbf x)=\mathbf w^\top\mathbf x$.
- **Doğrusal olmayan:** $\tilde R_{\mathbf w}(\mathbf x)=\text{SinirAğı}(\mathbf x;\mathbf w)$.

Öznitelikler iki biçimde kurulabilir. **Ayrık** modelde her kolun kendi ağırlık vektörü vardır ($r=\theta_a^\top\mathbf x_s$) ve kollar arasında bilgi paylaşılmaz. **Ortak** modelde tek bir $\mathbf w$, durum ve eylem özniteliklerinin birleşimi (ve çarpımları) üzerinde tanımlanır. Yeni bir reklam, benzer öznitelikli reklamlardan öğrenilenleri hemen kullanır. Bu, sürekli yeni kolların eklendiği haber önerisinde kritik bir avantajdır (Li ve ark. 2010).

Keşif için yalnız bir nokta kestirim yetmez; ödülün **belirsizliği** de gerekir. Bayesçi doğrusal regresyon bunu kapalı biçimde verir.

:::türetim Bayesçi doğrusal regresyon
Önsel Gauss olsun, $P(\mathbf w)=\mathcal N(\mathbf w\mid\mathbf 0,\lambda^2I)\propto\exp(-\mathbf w^\top\mathbf w/2\lambda^2)$. Olabilirlik de Gauss olsun, $P(r\mid\mathbf x,\mathbf w)=\mathcal N(r\mid\mathbf w^\top\mathbf x,\sigma^2)$. Sonsal ikisinin çarpımıyla orantılıdır:
$$P(\mathbf w\mid r,\mathbf x)\propto\exp\Bigl(-\frac{\mathbf w^\top\mathbf w}{2\lambda^2}\Bigr)\exp\Bigl(-\frac{(r-\mathbf w^\top\mathbf x)^2}{2\sigma^2}\Bigr).$$
Üs, $\mathbf w$'de ikinci derecedendir, dolayısıyla sonsal da Gauss'tur. Kareyi tamamlayınca $\mathbf w^\top\mathbf w$'nin katsayısı $\tfrac12\Sigma^{-1}$'i, doğrusal terim $\Sigma^{-1}\boldsymbol\mu$'yü verir. $n$ gözlem ($X$'in satırları $\mathbf x_i^\top$, $\mathbf r$ ödüller) için:
$$\Sigma=\bigl(\sigma^{-2}X^\top X+\lambda^{-2}I\bigr)^{-1},\qquad\boldsymbol\mu=\sigma^{-2}\,\Sigma X^\top\mathbf r.$$
Kaynak tek gözlem için $\Sigma=(\sigma^{-2}\mathbf x\mathbf x^\top+\lambda^{-2}I)^{-1}$ ve $\boldsymbol\mu=\sigma^{-2}\Sigma\mathbf xr$ veriyor; $n$ gözlem bunların toplamıdır. $\boldsymbol\mu$, $\lambda^{-2}\sigma^2$ cezalı sırt regresyonunun çözümüyle aynıdır. Bayesçi yorumun katkısı $\Sigma$'dır.

Yeni bir $\mathbf x$'te ödülün **öngörü dağılımı** iki belirsizliği toplar:
$$P(r\mid\mathbf x)=\int\mathcal N(r\mid\mathbf w^\top\mathbf x,\sigma^2)\,\mathcal N(\mathbf w\mid\boldsymbol\mu,\Sigma)\,d\mathbf w=\mathcal N\bigl(r\mid\mathbf x^\top\boldsymbol\mu,\ \mathbf x^\top\Sigma\mathbf x+\sigma^2\bigr).$$
$\mathbf x^\top\Sigma\mathbf x$, ağırlıklar hakkındaki belirsizliktir ve veri arttıkça küçülür. $\sigma^2$, gürültüdür ve hiç küçülmez. Tek boyutta $\lambda^2=\sigma^2=1$, $x=2$, $r=3$ ile $\Sigma=1/(4+1)=0.2$ ve $\mu=0.2\cdot2\cdot3=1.2$ olur. $x=1$'deki öngörü varyansı $0.2+1=1.2$'dir.
:::

:::uyarı Kaynak düzeltmesi (lect_11): öngörü dağılımı
Slayt öngörü dağılımını $\mathcal N(r\mid\sigma^2\mathbf x^\top\boldsymbol\mu,\ \mathbf x^\top\Sigma\mathbf x)$ olarak veriyor ve UCB ile Thompson formüllerinde de aynı ifadeyi kullanıyor. İki hata vardır:

1. $\boldsymbol\mu=\sigma^{-2}\Sigma\mathbf xr$ tanımı $\sigma^{-2}$'yi zaten içerir. Ortalama $\mathbf x^\top\boldsymbol\mu$'dür; başına ikinci bir $\sigma^2$ çarpanı gelmez.
2. Gelecekteki **ödülün** varyansı gürültüyü de içerir: $\mathbf x^\top\Sigma\mathbf x+\sigma^2$. Slayttaki $\mathbf x^\top\Sigma\mathbf x$, ortalama ödülün ($\mathbf w^\top\mathbf x$) varyansıdır. UCB ve Thompson için doğru nicelik de budur, çünkü belirsizliği azaltılabilecek olan ortalamadır. Ama "ödülün öngörü dağılımı" diye yazıldığında $\sigma^2$ eklenmelidir.
:::

<figure class="fig wide"><svg viewBox="0 0 640 254" role="img" aria-label="Bayesçi doğrusal regresyonun öngörü dağılımı ve LinUCB üst sınırı"><path d="M60,224.0 L460,224.0" class="mute"/><text x="54" y="228.0" text-anchor="end" class="sm">−2</text><path d="M60,179.55555555555554 L460,179.55555555555554" class="mute"/><text x="54" y="183.55555555555554" text-anchor="end" class="sm">−1</text><path d="M60,135.11111111111111 L460,135.11111111111111" class="mute"/><text x="54" y="139.11111111111111" text-anchor="end" class="sm">0</text><path d="M60,90.66666666666669 L460,90.66666666666669" class="mute"/><text x="54" y="94.66666666666669" text-anchor="end" class="sm">1</text><path d="M60,46.22222222222223 L460,46.22222222222223" class="mute"/><text x="54" y="50.22222222222223" text-anchor="end" class="sm">2</text><text x="60.0" y="240" text-anchor="middle" class="sm">−1.5</text><text x="126.66666666666666" y="240" text-anchor="middle" class="sm">−1</text><text x="193.33333333333331" y="240" text-anchor="middle" class="sm">−0.5</text><text x="260.0" y="240" text-anchor="middle" class="sm">0</text><text x="326.66666666666663" y="240" text-anchor="middle" class="sm">0.5</text><text x="393.33333333333337" y="240" text-anchor="middle" class="sm">1</text><text x="460.0" y="240" text-anchor="middle" class="sm">1.5</text><path d="M60.0,115.5 L63.3,115.5 L66.7,115.5 L70.0,115.5 L73.3,115.5 L76.7,115.4 L80.0,115.4 L83.3,115.4 L86.7,115.3 L90.0,115.3 L93.3,115.2 L96.7,115.2 L100.0,115.1 L103.3,115.0 L106.7,114.9 L110.0,114.9 L113.3,114.8 L116.7,114.7 L120.0,114.5 L123.3,114.4 L126.7,114.3 L130.0,114.1 L133.3,114.0 L136.7,113.8 L140.0,113.7 L143.3,113.5 L146.7,113.3 L150.0,113.1 L153.3,112.9 L156.7,112.6 L160.0,112.4 L163.3,112.1 L166.7,111.9 L170.0,111.6 L173.3,111.3 L176.7,111.0 L180.0,110.7 L183.3,110.4 L186.7,110.0 L190.0,109.6 L193.3,109.3 L196.7,108.9 L200.0,108.5 L203.3,108.0 L206.7,107.6 L210.0,107.1 L213.3,106.7 L216.7,106.2 L220.0,105.7 L223.3,105.2 L226.7,104.6 L230.0,104.1 L233.3,103.5 L236.7,102.9 L240.0,102.3 L243.3,101.7 L246.7,101.1 L250.0,100.5 L253.3,99.8 L256.7,99.1 L260.0,98.5 L263.3,97.8 L266.7,97.0 L270.0,96.3 L273.3,95.6 L276.7,94.8 L280.0,94.1 L283.3,93.3 L286.7,92.5 L290.0,91.7 L293.3,90.9 L296.7,90.1 L300.0,89.2 L303.3,88.4 L306.7,87.5 L310.0,86.7 L313.3,85.8 L316.7,84.9 L320.0,84.0 L323.3,83.1 L326.7,82.2 L330.0,81.3 L333.3,80.4 L336.7,79.5 L340.0,78.5 L343.3,77.6 L346.7,76.6 L350.0,75.7 L353.3,74.7 L356.7,73.8 L360.0,72.8 L363.3,71.8 L366.7,70.8 L370.0,69.8 L373.3,68.8 L376.7,67.8 L380.0,66.8 L383.3,65.8 L386.7,64.8 L390.0,63.8 L393.3,62.8 L396.7,61.7 L400.0,60.7 L403.3,59.7 L406.7,58.6 L410.0,57.6 L413.3,56.5 L416.7,55.5 L420.0,54.4 L423.3,53.4 L426.7,52.3 L430.0,51.3 L433.3,50.2 L436.7,49.1 L440.0,48.1 L443.3,47.0 L446.7,45.9 L450.0,44.8 L453.3,43.8 L456.7,42.7 L460.0,41.6 L460.0,156.0 L456.7,155.9 L453.3,155.8 L450.0,155.7 L446.7,155.7 L443.3,155.6 L440.0,155.5 L436.7,155.5 L433.3,155.4 L430.0,155.3 L426.7,155.3 L423.3,155.2 L420.0,155.2 L416.7,155.1 L413.3,155.1 L410.0,155.0 L406.7,155.0 L403.3,155.0 L400.0,154.9 L396.7,154.9 L393.3,154.9 L390.0,154.8 L386.7,154.8 L383.3,154.8 L380.0,154.8 L376.7,154.8 L373.3,154.8 L370.0,154.8 L366.7,154.8 L363.3,154.8 L360.0,154.9 L356.7,154.9 L353.3,154.9 L350.0,154.9 L346.7,155.0 L343.3,155.0 L340.0,155.1 L336.7,155.2 L333.3,155.2 L330.0,155.3 L326.7,155.4 L323.3,155.5 L320.0,155.6 L316.7,155.7 L313.3,155.8 L310.0,156.0 L306.7,156.1 L303.3,156.3 L300.0,156.4 L296.7,156.6 L293.3,156.8 L290.0,157.0 L286.7,157.2 L283.3,157.4 L280.0,157.6 L276.7,157.8 L273.3,158.1 L270.0,158.4 L266.7,158.6 L263.3,158.9 L260.0,159.2 L256.7,159.5 L253.3,159.9 L250.0,160.2 L246.7,160.6 L243.3,161.0 L240.0,161.4 L236.7,161.8 L233.3,162.2 L230.0,162.6 L226.7,163.1 L223.3,163.5 L220.0,164.0 L216.7,164.5 L213.3,165.0 L210.0,165.6 L206.7,166.1 L203.3,166.7 L200.0,167.3 L196.7,167.9 L193.3,168.5 L190.0,169.1 L186.7,169.7 L183.3,170.4 L180.0,171.0 L176.7,171.7 L173.3,172.4 L170.0,173.1 L166.7,173.9 L163.3,174.6 L160.0,175.3 L156.7,176.1 L153.3,176.9 L150.0,177.7 L146.7,178.5 L143.3,179.3 L140.0,180.1 L136.7,180.9 L133.3,181.8 L130.0,182.6 L126.7,183.5 L123.3,184.3 L120.0,185.2 L116.7,186.1 L113.3,187.0 L110.0,187.9 L106.7,188.8 L103.3,189.7 L100.0,190.7 L96.7,191.6 L93.3,192.5 L90.0,193.5 L86.7,194.5 L83.3,195.4 L80.0,196.4 L76.7,197.4 L73.3,198.3 L70.0,199.3 L66.7,200.3 L63.3,201.3 L60.0,202.3 Z" class="soft-a"/><path d="M60.0,158.9 L63.3,158.4 L66.7,157.9 L70.0,157.4 L73.3,156.9 L76.7,156.4 L80.0,155.9 L83.3,155.4 L86.7,154.9 L90.0,154.4 L93.3,153.9 L96.7,153.4 L100.0,152.9 L103.3,152.4 L106.7,151.9 L110.0,151.4 L113.3,150.9 L116.7,150.4 L120.0,149.9 L123.3,149.4 L126.7,148.9 L130.0,148.4 L133.3,147.9 L136.7,147.4 L140.0,146.9 L143.3,146.4 L146.7,145.9 L150.0,145.4 L153.3,144.9 L156.7,144.4 L160.0,143.9 L163.3,143.4 L166.7,142.9 L170.0,142.4 L173.3,141.9 L176.7,141.4 L180.0,140.9 L183.3,140.4 L186.7,139.9 L190.0,139.4 L193.3,138.9 L196.7,138.4 L200.0,137.9 L203.3,137.4 L206.7,136.9 L210.0,136.4 L213.3,135.9 L216.7,135.4 L220.0,134.9 L223.3,134.4 L226.7,133.9 L230.0,133.4 L233.3,132.9 L236.7,132.3 L240.0,131.8 L243.3,131.3 L246.7,130.8 L250.0,130.3 L253.3,129.8 L256.7,129.3 L260.0,128.8 L263.3,128.3 L266.7,127.8 L270.0,127.3 L273.3,126.8 L276.7,126.3 L280.0,125.8 L283.3,125.3 L286.7,124.8 L290.0,124.3 L293.3,123.8 L296.7,123.3 L300.0,122.8 L303.3,122.3 L306.7,121.8 L310.0,121.3 L313.3,120.8 L316.7,120.3 L320.0,119.8 L323.3,119.3 L326.7,118.8 L330.0,118.3 L333.3,117.8 L336.7,117.3 L340.0,116.8 L343.3,116.3 L346.7,115.8 L350.0,115.3 L353.3,114.8 L356.7,114.3 L360.0,113.8 L363.3,113.3 L366.7,112.8 L370.0,112.3 L373.3,111.8 L376.7,111.3 L380.0,110.8 L383.3,110.3 L386.7,109.8 L390.0,109.3 L393.3,108.8 L396.7,108.3 L400.0,107.8 L403.3,107.3 L406.7,106.8 L410.0,106.3 L413.3,105.8 L416.7,105.3 L420.0,104.8 L423.3,104.3 L426.7,103.8 L430.0,103.3 L433.3,102.8 L436.7,102.3 L440.0,101.8 L443.3,101.3 L446.7,100.8 L450.0,100.3 L453.3,99.8 L456.7,99.3 L460.0,98.8" class="ca w2"/><path d="M60.0,118.5 L63.3,118.8 L66.7,119.0 L70.0,119.2 L73.3,119.5 L76.7,119.7 L80.0,120.0 L83.3,120.2 L86.7,120.4 L90.0,120.7 L93.3,120.9 L96.7,121.1 L100.0,121.3 L103.3,121.5 L106.7,121.8 L110.0,122.0 L113.3,122.2 L116.7,122.4 L120.0,122.5 L123.3,122.7 L126.7,122.9 L130.0,123.1 L133.3,123.2 L136.7,123.4 L140.0,123.5 L143.3,123.7 L146.7,123.8 L150.0,123.9 L153.3,124.0 L156.7,124.1 L160.0,124.2 L163.3,124.2 L166.7,124.3 L170.0,124.3 L173.3,124.3 L176.7,124.3 L180.0,124.2 L183.3,124.1 L186.7,124.0 L190.0,123.9 L193.3,123.7 L196.7,123.5 L200.0,123.3 L203.3,123.0 L206.7,122.7 L210.0,122.3 L213.3,121.9 L216.7,121.4 L220.0,120.9 L223.3,120.4 L226.7,119.8 L230.0,119.1 L233.3,118.4 L236.7,117.7 L240.0,117.0 L243.3,116.2 L246.7,115.3 L250.0,114.4 L253.3,113.5 L256.7,112.6 L260.0,111.7 L263.3,110.7 L266.7,109.7 L270.0,108.7 L273.3,107.6 L276.7,106.6 L280.0,105.5 L283.3,104.4 L286.7,103.3 L290.0,102.2 L293.3,101.0 L296.7,99.9 L300.0,98.7 L303.3,97.6 L306.7,96.4 L310.0,95.3 L313.3,94.1 L316.7,92.9 L320.0,91.7 L323.3,90.5 L326.7,89.3 L330.0,88.1 L333.3,86.9 L336.7,85.7 L340.0,84.4 L343.3,83.2 L346.7,82.0 L350.0,80.8 L353.3,79.5 L356.7,78.3 L360.0,77.0 L363.3,75.8 L366.7,74.6 L370.0,73.3 L373.3,72.1 L376.7,70.8 L380.0,69.6 L383.3,68.3 L386.7,67.0 L390.0,65.8 L393.3,64.5 L396.7,63.3 L400.0,62.0 L403.3,60.7 L406.7,59.5 L410.0,58.2 L413.3,57.0 L416.7,55.7 L420.0,54.4 L423.3,53.1 L426.7,51.9 L430.0,50.6 L433.3,49.3 L436.7,48.1 L440.0,46.8 L443.3,45.5 L446.7,44.2 L450.0,43.0 L453.3,41.7 L456.7,40.4 L460.0,39.1" class="cb w2" stroke-dasharray="5 4"/><path d="M60.0,166.2 L63.3,165.6 L66.7,164.9 L70.0,164.2 L73.3,163.6 L76.7,162.9 L80.0,162.2 L83.3,161.6 L86.7,160.9 L90.0,160.2 L93.3,159.6 L96.7,158.9 L100.0,158.2 L103.3,157.6 L106.7,156.9 L110.0,156.2 L113.3,155.6 L116.7,154.9 L120.0,154.2 L123.3,153.6 L126.7,152.9 L130.0,152.2 L133.3,151.6 L136.7,150.9 L140.0,150.2 L143.3,149.6 L146.7,148.9 L150.0,148.2 L153.3,147.6 L156.7,146.9 L160.0,146.2 L163.3,145.6 L166.7,144.9 L170.0,144.2 L173.3,143.6 L176.7,142.9 L180.0,142.2 L183.3,141.6 L186.7,140.9 L190.0,140.2 L193.3,139.6 L196.7,138.9 L200.0,138.2 L203.3,137.6 L206.7,136.9 L210.0,136.2 L213.3,135.6 L216.7,134.9 L220.0,134.2 L223.3,133.6 L226.7,132.9 L230.0,132.2 L233.3,131.6 L236.7,130.9 L240.0,130.2 L243.3,129.6 L246.7,128.9 L250.0,128.2 L253.3,127.6 L256.7,126.9 L260.0,126.2 L263.3,125.6 L266.7,124.9 L270.0,124.2 L273.3,123.6 L276.7,122.9 L280.0,122.2 L283.3,121.6 L286.7,120.9 L290.0,120.2 L293.3,119.6 L296.7,118.9 L300.0,118.2 L303.3,117.6 L306.7,116.9 L310.0,116.2 L313.3,115.6 L316.7,114.9 L320.0,114.2 L323.3,113.6 L326.7,112.9 L330.0,112.2 L333.3,111.6 L336.7,110.9 L340.0,110.2 L343.3,109.6 L346.7,108.9 L350.0,108.2 L353.3,107.6 L356.7,106.9 L360.0,106.2 L363.3,105.6 L366.7,104.9 L370.0,104.2 L373.3,103.6 L376.7,102.9 L380.0,102.2 L383.3,101.6 L386.7,100.9 L390.0,100.2 L393.3,99.6 L396.7,98.9 L400.0,98.2 L403.3,97.6 L406.7,96.9 L410.0,96.2 L413.3,95.6 L416.7,94.9 L420.0,94.2 L423.3,93.6 L426.7,92.9 L430.0,92.2 L433.3,91.6 L436.7,90.9 L440.0,90.2 L443.3,89.6 L446.7,88.9 L450.0,88.2 L453.3,87.6 L456.7,86.9 L460.0,86.2" class="mute w2"/><circle cx="140.0" cy="158.9" r="4" class="cb-f"/><circle cx="180.0" cy="144.6" r="4" class="cb-f"/><circle cx="206.7" cy="114.7" r="4" class="cb-f"/><circle cx="273.3" cy="114.8" r="4" class="cb-f"/><circle cx="286.7" cy="142.8" r="4" class="cb-f"/><text x="60" y="14" class="sm">Bayesçi doğrusal regresyon, φ(x) = (1, x), λ = 1.0, σ = 0.3, 5 gözlem</text><text x="480" y="34" class="sm">mavi: öngörü ortalaması xᵀμ</text><text x="480" y="50" class="sm">gölge: ± 2 öngörü std</text><text x="480" y="66" class="sm">  √(xᵀΣx + σ²)</text><text x="480" y="82" class="sm">turuncu kesikli: LinUCB</text><text x="480" y="98" class="sm">  xᵀμ + c√(xᵀΣx), c = 2.36</text><text x="480" y="114" class="sm">gri: gerçek ödül</text><text x="480" y="130" class="sm">noktalar: gözlemler</text><text x="480" y="146" class="sm"></text><text x="480" y="162" class="sm">Gözlemlerden uzakta</text><text x="480" y="178" class="sm">(x > 0.5) belirsizlik ve</text><text x="480" y="194" class="sm">üst sınır büyür: keşif</text><text x="480" y="210" class="sm">oraya yönelir.</text></svg><figcaption>Şekil 1 — Bağlamlı haydutun ödül modeli: sonsal N(μ, Σ) ile her bağlamda bir öngörü ortalaması ve belirsizlik çıkar. Belirsizlik veriden uzaklaştıkça büyür; LinUCB bu belirsizliği bonus olarak ekler, Thompson sonsaldan örnek çeker (hesaplandı).</figcaption></figure>

### LinUCB ve doğrusal Thompson örneklemesi {#linucb}

Öngörü dağılımından iki keşif kuralı çıkar.

**LinUCB** (Li ve ark. 2010) her eylem için ortalama ödülün bir üst güven sınırını hesaplar ve en büyüğünü seçer:
$$\text{GüvenSınırı}(a)=\mathbf x^\top\boldsymbol\mu+c\sqrt{\mathbf x^\top\Sigma\mathbf x},\qquad c=1+\sqrt{\ln(2/\delta)/2}.$$
$\delta=0.05$ ile $c=2.36$ olur. Bonus, M11.7'deki $\sqrt{2\ln n/n_a}$'nın çok boyutlu karşılığıdır. Bağlamın, verinin az kapsadığı yönlerinde büyüktür.

```text
LinUCB(h):
    V ← 0;  P(w) = N(0, λ²I)  (μ ← 0, Σ ← λ²I)
    n = h olana dek:
        x_s durumunu al
        her eylem için x = (x_s, x_a):  GüvenSınırı(a) ← xᵀμ + c √(xᵀΣx)
        a* ← argmax_a GüvenSınırı(a);  uygula, r'yi al;  V ← V + r
        x = (x_s, x_{a*}) ve r ile μ ve Σ'yı güncelle
    döndür V
```

**Doğrusal Thompson örneklemesi** (Agrawal ve Goyal 2013) sonsaldan **tek** bir ağırlık vektörü çeker, $\tilde{\mathbf w}\sim\mathcal N(\boldsymbol\mu,\Sigma)$, ve $\arg\max_a\mathbf x^\top\tilde{\mathbf w}$'yi seçer. Kaynaktaki sözde kod her eylem için ayrı ayrı $R_i(a)$ örnekliyor. Ortak bir modelde eylemlerin ödülleri aynı $\mathbf w$'ye bağlıdır ve korelelidir. Her eylem için bağımsız örneklemek bu ilişkiyi kaybettirir. Doğru yol tek bir $\tilde{\mathbf w}$ çekip bütün eylemleri onunla değerlendirmektir. Ayrık modelde her kolun kendi sonsalından birer örnek çekilir.

:::türetim Sherman–Morrison: güncellemeyi $O(d^2)$'ye indirmek
Her gözlemden sonra $\Sigma^{-1}\leftarrow\Sigma^{-1}+\sigma^{-2}\mathbf x\mathbf x^\top$ olur. Tersi baştan almak $O(d^3)$ ister. Rank-1 güncellemenin tersi kapalı biçimde bilinir:
$$(A+\mathbf u\mathbf v^\top)^{-1}=A^{-1}-\frac{A^{-1}\mathbf u\mathbf v^\top A^{-1}}{1+\mathbf v^\top A^{-1}\mathbf u}.$$
Doğrulamak için sağ taraf $A+\mathbf u\mathbf v^\top$ ile çarpılır; $\mathbf v^\top A^{-1}\mathbf u$ bir sayı olduğu için terimler sadeleşir ve $I$ kalır. $A^{-1}\mathbf u$ bir matris–vektör çarpımıdır ($O(d^2)$) ve dış çarpım da $O(d^2)$'dir. $A^{-1}=I_2$, $\mathbf u=\mathbf v=(1,1)$ için payda $1+2=3$ olur ve sonuç $\begin{bmatrix}2/3&-1/3\\-1/3&2/3\end{bmatrix}$'tür. $d=100$ özniteliğin bulunduğu bir sistemde bu, adım başına $10^6$ yerine $10^4$ mertebesinde işlem demektir.
:::

**Açgözlü yeter mi?** Bağlamlar çeşitliyse, açgözlü seçim bile her kolu farklı bağlamlarda deneyerek yeterli bilgi toplayabilir. Bastani ve ark. (2021) bağlam dağılımı her yönü kapsadığında açgözlünün de iyi pişmanlık verebildiğini gösterdi. Aşağıdaki deneyde bağlamlar $\mathcal N(0,I)$'den geldiği için bağlamlı açgözlü de LinUCB'ye yakın sonuç veriyor. Bağlamı hiç kullanmayan UCB1 ise doğrusal pişmanlık yapıyor: her kolun ortalama ödülü sıfıra yakındır, ama doğru kol bağlamdan bağlama değişir.

::lab l1

### Doğrusal olmayan ödüller {#neural}

Ödül doğrusal değilse sinir ağı kullanılır. Ama sinir ağının sonsalı kapalı biçimde hesaplanamaz. Yaygın bir çözüm **NeuralLinear**'dır (Riquelme ve ark. 2018). Ağ, olağan regresyonla eğitilir ve son gizli katmanı bir öznitelik vektörü olarak kullanılır. Bu özniteliklerin üzerine Bayesçi doğrusal regresyon kurulur ve LinUCB ya da Thompson uygulanır. Temsil derin ağdan, belirsizlik son katmanın kapalı biçimli sonsalından gelir. Karşılaştırmalı çalışmalarda bu basit yöntem daha karmaşık Bayesçi sinir ağı yaklaşımlarıyla rekabet eder.

### Kayıtlı veriden değerlendirme {#ope}

Bir sistemde yeni bir politikayı canlı denemek pahalı ve risklidir. Elde ise eski politikanın kayıtları vardır: $(\mathbf x_i,a_i,r_i,p_i)$. $p_i$, kayıt politikasının o eylemi seçme olasılığıdır (eğilim). Soru şudur: yeni bir $\pi$ politikası canlıya alınsaydı ortalama ödül ne olurdu? Bu **politika dışı değerlendirmedir** (OPE). Zorluğu, $\pi$'nin seçeceği eylemin ödülünün çoğu kayıtta gözlenmemiş olmasıdır.

- **Doğrudan yöntem (DM):** kayıtlardan bir ödül modeli $\hat r(\mathbf x,a)$ öğrenilir ve $\hat V_{\text{DM}}=\frac1n\sum_i\hat r(\mathbf x_i,\pi(\mathbf x_i))$ hesaplanır. Varyansı düşüktür, ama model yanlışsa yanlıdır.
- **Ters eğilim ağırlıklandırma (IPS):** $\hat V_{\text{IPS}}=\frac1n\sum_i\frac{\mathbb 1[a_i=\pi(\mathbf x_i)]}{p_i}r_i$. Kayıt politikasının nadiren seçtiği eylemler büyük ağırlık alır.
- **Çift sağlam (DR):** $\hat V_{\text{DR}}=\frac1n\sum_i\Bigl[\hat r(\mathbf x_i,\pi(\mathbf x_i))+\frac{\mathbb 1[a_i=\pi(\mathbf x_i)]}{p_i}\bigl(r_i-\hat r(\mathbf x_i,a_i)\bigr)\Bigr]$.

:::türetim IPS yansızdır; DR iki kez sağlamdır
$a\sim p(\cdot\mid\mathbf x)$ iken beklenti alınır:
$$E\Bigl[\frac{\mathbb 1[a=\pi(\mathbf x)]}{p(a\mid\mathbf x)}r\Bigr]=\sum_ap(a\mid\mathbf x)\frac{\mathbb 1[a=\pi(\mathbf x)]}{p(a\mid\mathbf x)}E[r\mid\mathbf x,a]=E[r\mid\mathbf x,\pi(\mathbf x)].$$
Bunun için $\pi$'nin seçebileceği her eylemde $p>0$ olmalıdır (destek koşulu). Varyans $1/p$ ile büyür. DR'de düzeltme teriminin beklentisi $E[r\mid\mathbf x,\pi(\mathbf x)]-\hat r(\mathbf x,\pi(\mathbf x))$'dir. Eğilimler doğruysa DR yansızdır; model ne kadar yanlış olursa olsun düzeltme terimi açığı kapatır. Model doğruysa düzeltmenin beklentisi sıfırdır ve DR, eğilimler yanlış olsa bile yansızdır. "Çift sağlam" adı buradan gelir. Varyansı genellikle IPS'ten düşüktür, çünkü ağırlıklandırılan şey ödülün tamamı değil, modelin açıklayamadığı artıktır.
:::

**Öz-normalize IPS** (SNIPS) ağırlıkların toplamına böler. Biraz yanlıdır, ama varyansı ve aşırı değerlere duyarlılığı daha düşüktür. Haber önerisinde Li ve ark. (2011), rastgele kayıt politikasıyla toplanmış veride **tekrar oynatma** yöntemini kullandı: yalnız kayıttaki eylemin yeni politikanın seçimiyle çakıştığı adımlar sayılır. Bu, düzgün eğilimli IPS'in özel bir hâlidir.

::lab l2

### Endüstride {#endustri}

Bağlamlı haydutlar kişiselleştirilmiş reklamda ve web içeriğinde yaygındır. Microsoft, MSN haberlerinde bağlamlı haydutlara geçtikten sonra tıklanma oranında %26 artış bildirdi (Microsoft Research Blog, Ağustos 2017). Bu tür sistemlerde öğrenme döngüsünün dört parçası birlikte kurulur:

1. **Keşfeden politika:** eğilimleri kaydedilen ε-açgözlü, LinUCB ya da Thompson.
2. **Kayıt:** bağlam, eylem, ödül ve eğilim.
3. **Çevrimdışı öğrenme:** ödül modeli ve yeni politika.
4. **Çevrimdışı değerlendirme:** IPS, DR ya da tekrar oynatma ile, canlıya almadan önce.

### Özet {#ozet}

- Bağlamlı haydut, durumlu ama geçişsiz bir problemdir. Bağlamlar bağımsız gelir, her adım tek adımlık bir karardır. Amaç $\pi:\mathbf x_s\to a$, $E[r\mid\mathbf x_s,\mathbf x_a]$'yı en çoklar.
- Ödül regresyonla yaklaşılır: doğrusal, ayrık ya da ortak öznitelikler; doğrusal olmayanda sinir ağı ve NeuralLinear.
- Bayesçi doğrusal regresyon: $\Sigma=(\sigma^{-2}X^\top X+\lambda^{-2}I)^{-1}$, $\boldsymbol\mu=\sigma^{-2}\Sigma X^\top\mathbf r$. Öngörü dağılımı $\mathcal N(\mathbf x^\top\boldsymbol\mu,\mathbf x^\top\Sigma\mathbf x+\sigma^2)$'dir; slayttaki ek $\sigma^2$ çarpanı ve eksik gürültü terimi düzeltildi.
- LinUCB: $\mathbf x^\top\boldsymbol\mu+c\sqrt{\mathbf x^\top\Sigma\mathbf x}$. Doğrusal Thompson: tek bir $\tilde{\mathbf w}\sim\mathcal N(\boldsymbol\mu,\Sigma)$ ile bütün eylemleri değerlendirir. Sherman–Morrison güncellemeyi $O(d^2)$ yapar.
- Bağlam çeşitliyse açgözlü de iyi çalışabilir. Bağlamı yok saymak doğrusal pişmanlık getirir.
- Politika dışı değerlendirme: DM yanlı ama düşük varyanslı, IPS yansız ama varyanslı (destek koşulu), DR iki koşuldan biri yeterli olunca yansızdır.
