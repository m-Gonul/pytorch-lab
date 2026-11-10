### Video: uzay-zaman sinyali {#giris}

"Video" sözcüğü hem durağan görüntüleri hem de zamanla değişen görüntüleri kapsar. Durağan bir görüntü, zamana göre sabit bir uzamsal yoğunluk dağılımıdır. Zamanla değişen bir görüntüde ise uzamsal desen zamanla değişir. Bu yüzden video bir **uzay-zaman yoğunluk deseni** olarak yazılır: $s_c(x_1,x_2,t)$. Burada $x_1,x_2$ uzamsal, $t$ zamansal değişkendir; $c$ alt indisi sürekli sinyali gösterir. Video uzun süre analog kaydedilip iletildi. Ancak analog sinyalleri iletmek ve işlemek zordur, bugün ise sayısal bilgisayarlar her biçimde erişilebilir: akıllı telefonlar, gömülü endüstriyel ve askerî bilgisayarlar, dizüstü ve masaüstü bilgisayarlar, sunucular ve bulut. Pratikte bütün video işleme sayısal alanda yapılıyor ve bu modül yalnız sayısal videoyla ilgileniyor.

Sayısal bir videoyu betimleyen parametreler şunlardır:

- **zamansal çözünürlük:** saniyedeki kare sayısı, FPS;
- **uzamsal çözünürlük:** 1080p, 720p, …;
- **sıkıştırılmış ya da sıkıştırılmamış** olması;
- **sıkıştırma biçimi:** video ve ses codec'i;
- **renk uzayı.**

Düşük FPS'te hareket kesik kesik görünür; yüksek FPS'te aynı hareket akıcıdır.

Sayısal video işleme görevlerinin listesi uzundur:

- optik akış (hareket) kestirimi,
- kare hızı dönüşümü,
- kare ölçekleme,
- kontrast, renk iyileştirme ve gürültü süzme,
- tek ve çok kareden çözünürlük artırma,
- stabilizasyon,
- arka plan çıkarma (M9.18),
- sıkıştırma.

Bu modül hareket kestirimini ve onun üç uygulamasını işliyor: zamansal gürültü giderme, kare hızı artırma ve stabilizasyon.

### Hareket: tek kare ve çok kare yöntemler {#hareket}

Video sinyallerine anlatım gücünü veren, gerçek dünyadaki hareketi yakalayabilmeleridir. Bütün kareler aynı olsaydı sinyal durağan bir görüntüden farksız olurdu. Videoyu ilginç kılan hareket, onu işlemenin de ana zorluğudur; buna sıkıştırma gibi kaynaklardan gelen bozulmalar eklenir.

Video işleme yöntemleri iki gruba ayrılır:

- **Tek kare yöntemleri:** her kareyi diğerlerinden bağımsız işler. Önceki modüllerdeki bütün görüntü işleme yöntemleri bu biçimde uygulanabilir.
- **Çok kare yöntemleri:** her çıkış karesini birden çok giriş karesinden hesaplar. Bunlar iki türdür:
    - **hareket duyarlı** (motion aware/adaptive): her pikselde hareket olup olmadığına göre davranışını değiştirir;
    - **hareket telafili** (motion compensated): hareketi kestirip kareleri hizaladıktan sonra birleştirir.

Çok kare yöntemleri genellikle tek kare yöntemlerinden daha iyi sonuç verir, ama hareket bilgisi ister. Hareket bilgisini kullanmanın iki yolu vardır:

- **Hareket tespiti:** her piksel için "hareket var mı?" sorusunu yanıtlar. Basittir ve az hesap ister.
- **Hareket kestirimi:** ardışık kareleri ilişkilendiren hareket vektörlerini bulur. Hesap açısından pahalıdır.

Video işlemenin "kutsal kâsesi" sayılır: yüksek kaliteli, yüksek çözünürlüklü hareket vektörleri varsa her video işleme görevi yüksek başarımla yapılabilir.

### Zamansal gürültü giderme {#tnr}

Uzamsal gürültü süzgeçleri (M9.5, M9.12) ortalama alır; farkları nasıl ortaladıklarındadır:

- uyarlamalı (ikili süzgeç) ya da sabit (Gauss);
- yerel olmayan (NLM) ya da yerel;
- ağırlıklı (Gauss) ya da ağırlıksız (kutu).

Videoda ortalama **zaman** ekseni boyunca da alınabilir: önceki ve sonraki karelerde benzer pikseller aranır. Üç düzey vardır:

1. **Temel:** aynı piksel konumunu birkaç kare boyunca ortalamak.
2. **Gelişmiş:** her pikselde hareketi tespit etmek ve hareket varsa o pikseli ortalamanın dışında bırakmak.
3. **Güncel (SoTA):** hareketi kestirmek, kareleri hareket yörüngesi boyunca hizalamak ve ağırlıklı ortalama almak. Ağırlıklar hareket vektörlerinin güvenilirliğine bağlıdır.

Ders görselindeki hareket telafili ortalamada piksel $(\mathbf n,k)$, komşu karelerde hareket yörüngesi üzerindeki $g(\mathbf n,k\pm1)$, $g(\mathbf n,k\pm2)$ değerleriyle ortalanır. Uzay-zaman pencereleri de kullanılır: bir $3\times3$ uzamsal komşuluk, yörünge boyunca üç karede birlikte ortalanır.

:::türetim Zamansal ortalama ne kazandırır, ne kaybettirir?
Hareketsiz bir pikselin $N$ karedeki gözlemleri $g_k=f+n_k$ olsun; $n_k$ bağımsız, ortalaması sıfır ve varyansı $\sigma^2$ olan gürültü. Ortalama $\bar g=f+\frac1N\sum n_k$'nin varyansı
$$\operatorname{Var}(\bar g)=\frac1{N^2}\sum_k\operatorname{Var}(n_k)=\frac{\sigma^2}N,$$
yani gürültü std'si $\sigma/\sqrt N$'e iner: $\sigma=10$ ve 4 kare için 5. Uzamsal ortalamanın aksine kenarlar bulanıklaşmaz, çünkü ortalanan değerler aynı sahne noktasına aittir.

Varsayım bozulursa, yani piksel kareler arasında hareket ediyorsa, $\bar g$ farklı sahne noktalarının ortalamasıdır. Hareket eden nesnenin arkasında silik bir iz (**hayalet**, ghosting) kalır. Hareket duyarlı yöntem bu pikselleri ortalamadan çıkarır; o piksellerde gürültü azalmaz ama iz de oluşmaz. Hareket telafili yöntem ise yörüngeyi izleyerek her pikselde $\sigma/\sqrt N$'e yaklaşır, tabii hareket doğru kestirilmişse.
:::

Hareket tespitinin en basit biçimi bir eşiktir: $|g_k(\mathbf n)-g_{k-1}(\mathbf n)|>\tau$ ise pikselde hareket var sayılır. $\tau$ gürültünün birkaç katı seçilmelidir. İki gürültülü pikselin farkının std'si $\sqrt2\sigma$ olduğu için $\tau\approx3\sqrt2\sigma$ yaygın bir başlangıçtır.

### Optik akış: tanım ve kötü konumlanmışlık {#optik-akis}

Video kareleri 3-B gerçek dünyanın 2-B görüntü düzlemine izdüşümleridir. Gözlenen 2-B hareket de 3-B hareketin izdüşümüdür. Video işleme görevlerinin çoğunda görüntü düzlemindeki 2-B hareket yeterlidir.

Bir **karşılık vektörü** (yer değiştirme vektörü, hareket vektörü), bir piksel bir kareden ötekine geçerken görüntü koordinatlarındaki satır ve sütun kaymalarıdır. Probleme göre ardışık kareler arasındaki vektörler (iki zaman yönünde) ya da aralarında başka kareler bulunan kareler arasındaki vektörler gerekir. **Optik akış** ise görüntü koordinatlarının zamana göre değişim hızıdır: $(dx/dt,dy/dt)$. Türetimlerde hız tanımı matematiksel olarak daha temizdir. Pratikte sayısal video sürekli olmadığı için $dt=1$ alınır ve optik akış ile yer değiştirme vektörü birbirinin yerine kullanılır.

Hareket kestirimi **kötü konumlanmış** (ill-posed) bir problemdir. Çözümün tek ve kararlı olması için düzenleme gerekir, örneğin parametreli hareket modelleri. Üç temel zorluk vardır:

- **Örtme** (occlusion): bir karede görünen bölge ötekinde bir nesnenin arkasında kalır, karşılığı yoktur. Açığa çıkan (disocclusion) bölge için de aynısı geçerlidir.
- **Açıklık** (aperture): küçük bir pencereden düz bir kenara bakınca yalnız kenara dik bileşen ölçülebilir. Bunu aşağıda optik akış denkleminden türeteceğiz.
- **Paralaks** (parallax): kamera ötelenirken yakın nesneler uzaktakilerden daha çok yer değiştirir. Tek bir küresel model sahnenin tamamını açıklayamaz.

**Hareket modelleri.** Modeller, bir hareket tanımının geçerli olduğu bölgeye (destek bölgesine) göre sınıflanır:

1. **Görüntü düzeyi:** bütün kare için tek bir model ya da parametre kümesi, örneğin kamera dönmesi ya da yakınlaşması.
2. **Piksel düzeyi:** her piksele bir vektör (yoğun optik akış).
3. **Blok düzeyi:** her 8 × 8 ya da 16 × 16 bloğa bir vektör.
4. **Nesne düzeyi:** her nesneye bir vektör.

**Kestirilen hareket ve gerçek hareket.** Gerçek hareket vektörlerine her zaman ihtiyaç var mıdır? Yoksa yakın eşleşen herhangi bir blok yeter mi? Bu göreve bağlıdır:

- **Sıkıştırma** için gerçek hareket gerekmez. Amaç artık (residual) enerjisini en aza indirmektir; tekrar eden bir dokuda yanlış ama iyi eşleşen bir blok da iş görür.
- **Kare enterpolasyonu ve süper çözünürlük** için gerçek hareket gerekir. Yanlış vektör, nesneyi ara karede yanlış yere koyar.

### Pratik algoritmalar: küresel hareket ve blok eşleme {#blok}

Gerçek videolarda hareket kestirimi çok zordur ve genellikle gerçek zamanda yapılması istenir. Gerçek zamanda tatmin edici çalışan üç aile vardır: küresel hareket kestirimi, blok eşleme ve düzenlemeli optik akış. Bazı fikirler yöntemden bağımsız olarak hepsine uygulanır:

- **hiyerarşik uygulama:** kabadan inceye, piramitle;
- **düzenleme:** komşu vektörlerin benzer olması;
- **yedek mekanizmalar** (fallback): güvenilmeyen vektör yerine güvenli bir varsayılan.

**Küresel hareket kestirimi.** Giriş karesine uygulandığında bir sonraki kareyi veren afin (ya da daha yüksek mertebeden) bir bükme matrisi bulunur; bükme, karenin hareketidir. Bu yaklaşım hareketsiz sahnelerde kamera hareketi için iyi çalışır: öteleme, yakınlaşma/uzaklaşma ve dönme. Kestirim M9.15'teki gibi yapılır: öznitelik eşleştirme, ardından RANSAC ile afin ya da homografi uydurma.

**Blok eşleme.** Video sıkıştırmada tercih edilen yöntemdir. $t_k$ anındaki karede $M\times N$'lik bir $B_m$ bloğu, $t_{k-1}$ karesinde bir $P$ arama alanındaki bütün aday bloklarla karşılaştırılır; en benzer adayın konum farkı bloğun hareket vektörüdür. Benzerlik ölçütleri:
$$\text{SAD}=\sum_{\mathbf x\in B}|B_1(\mathbf x)-B_0(\mathbf x+\mathbf v)|,\quad \text{MAD}=\text{SAD}/|B|,\quad \text{SSD}=\sum(\cdot)^2,\quad \text{MSE}=\text{SSD}/|B|.$$
Kontrast ve parlaklık düzeltmeli sürümler de kullanılır; örneğin her bloğun ortalaması çıkarılmış SAD. SAD donanımda en ucuz ölçüttür. SSD büyük farkları daha çok cezalandırır.

**Maliyet.** ±$R$ arama aralığında **tam arama** $(2R+1)^2$ adayı dener. 16 × 16 blok ve ±7 için bu 225 aday ve blok başına $225\cdot256=57\,600$ mutlak farktır. 1080p bir karede 8 160 blok vardır, yani kare başına $4.7\times10^8$ işlem. **Üç adımlı arama** (TSS; Koga ve ark. 1981) önce 4 piksel aralıklı 9 noktayı dener, en iyisinin çevresinde adımı 2'ye, sonra 1'e indirir: $1+3\cdot8=25$ aday. TSS, maliyet yüzeyinin en iyi eşleşmeye doğru tekdüze azaldığını varsayar. Dokulu bloklarda bu varsayım bozulur ve arama yerel minimuma takılabilir.

<figure class="fig wide"><svg viewBox="0 0 640 282" role="img" aria-label="Üç adımlı blok arama: ±7 penceresinde 4, 2 ve 1 adımla ziyaret edilen adaylar ve yol"><path d="M40,30 L40,255" class="mute"/><path d="M40,30 L265,30" class="mute"/><path d="M55,30 L55,255" class="mute"/><path d="M40,45 L265,45" class="mute"/><path d="M70,30 L70,255" class="mute"/><path d="M40,60 L265,60" class="mute"/><path d="M85,30 L85,255" class="mute"/><path d="M40,75 L265,75" class="mute"/><path d="M100,30 L100,255" class="mute"/><path d="M40,90 L265,90" class="mute"/><path d="M115,30 L115,255" class="mute"/><path d="M40,105 L265,105" class="mute"/><path d="M130,30 L130,255" class="mute"/><path d="M40,120 L265,120" class="mute"/><path d="M145,30 L145,255" class="mute"/><path d="M40,135 L265,135" class="mute"/><path d="M160,30 L160,255" class="mute"/><path d="M40,150 L265,150" class="mute"/><path d="M175,30 L175,255" class="mute"/><path d="M40,165 L265,165" class="mute"/><path d="M190,30 L190,255" class="mute"/><path d="M40,180 L265,180" class="mute"/><path d="M205,30 L205,255" class="mute"/><path d="M40,195 L265,195" class="mute"/><path d="M220,30 L220,255" class="mute"/><path d="M40,210 L265,210" class="mute"/><path d="M235,30 L235,255" class="mute"/><path d="M40,225 L265,225" class="mute"/><path d="M250,30 L250,255" class="mute"/><path d="M40,240 L265,240" class="mute"/><path d="M265,30 L265,255" class="mute"/><path d="M40,255 L265,255" class="mute"/><circle cx="152.5" cy="142.5" r="6" class="cb-f" opacity="0.85"/><circle cx="92.5" cy="82.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="92.5" cy="142.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="92.5" cy="202.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="152.5" cy="82.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="152.5" cy="202.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="212.5" cy="82.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="212.5" cy="142.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="212.5" cy="202.5" r="5.5" class="ca-f" opacity="0.85"/><circle cx="62.5" cy="172.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="62.5" cy="202.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="62.5" cy="232.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="92.5" cy="172.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="92.5" cy="232.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="122.5" cy="172.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="122.5" cy="202.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="122.5" cy="232.5" r="4.5" class="cp-f" opacity="0.85"/><circle cx="77.5" cy="187.5" r="3.5" class="cq-f" opacity="0.85"/><circle cx="77.5" cy="202.5" r="3.5" class="cq-f" opacity="0.85"/><circle cx="77.5" cy="217.5" r="3.5" class="cq-f" opacity="0.85"/><circle cx="92.5" cy="187.5" r="3.5" class="cq-f" opacity="0.85"/><circle cx="92.5" cy="217.5" r="3.5" class="cq-f" opacity="0.85"/><circle cx="107.5" cy="187.5" r="3.5" class="cq-f" opacity="0.85"/><circle cx="107.5" cy="202.5" r="3.5" class="cq-f" opacity="0.85"/><circle cx="107.5" cy="217.5" r="3.5" class="cq-f" opacity="0.85"/><path d="M152.5,142.5 L92.5,202.5 L92.5,202.5 L107.5,217.5" class="cb w2" fill="none"/><rect x="100.5" y="210.5" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5"/><text x="40" y="20" class="sm">arama penceresi ±7 (dikey eksen aşağı pozitif)</text><text x="300" y="50" class="sm">adımlar 4 → 2 → 1, her adımda 8 komşu</text><text x="300" y="72" class="sm">yol: (0, 0) → (-4, 4) → (-4, 4) → (-3, 5)</text><text x="300" y="94" class="sm">değerlendirilen aday: 25 (tam arama 225)</text><text x="300" y="116" class="sm">en iyi vektör (-3, 5) = hedef (kare)</text><text x="300" y="138" class="sm">maliyet yüzeyi tek tepeli olduğu için doğru buldu;</text><text x="300" y="160" class="sm">dokulu bloklarda yerel minimumlara takılabilir.</text><text x="300" y="182" class="sm">Benzerlik: SAD = Σ|B₁ − B₀|, MAD = SAD / B²,</text><text x="300" y="204" class="sm">SSD = Σ(B₁ − B₀)², ortalaması çıkarılmış SAD.</text><circle cx="306" cy="236" r="5" class="cb-f"/><text x="316" y="240" class="sm">başlangıç (0, 0)</text><circle cx="456" cy="236" r="5" class="ca-f"/><text x="466" y="240" class="sm">adım 4</text><circle cx="306" cy="256" r="5" class="cp-f"/><text x="316" y="260" class="sm">adım 2</text><circle cx="456" cy="256" r="5" class="cq-f"/><text x="466" y="260" class="sm">adım 1</text></svg><figcaption>Şekil 2 — Üç adımlı arama 25 adayla tam aramanın 225 adayının yerini tutar (hesaplandı; maliyet yüzeyi örnek olarak tek tepeli seçildi).</figcaption></figure>

### Optik akış denklemi ve açıklık problemi {#denklem}

Video sinyali $I(x,y,t)$ olsun. Temel varsayım **parlaklık sabitliğidir**: bir sahne noktası hareket ederken parlaklığı değişmez. Ayrık biçimde $I(x,y,t)=I(x+u,y+v,t+1)$, sürekli biçimde ise yörünge boyunca toplam türev sıfırdır: $dI/dt=0$.

:::türetim Optik akış denklemi
Zincir kuralıyla
$$\frac{dI}{dt}=\frac{\partial I}{\partial x}\frac{dx}{dt}+\frac{\partial I}{\partial y}\frac{dy}{dt}+\frac{\partial I}{\partial t}=0.$$
Aynı sonuca ayrık biçimden de ulaşılır. **Küçük hareket** varsayımıyla birinci mertebe Taylor açılımı
$$I(x+u,y+v,t+1)\approx I(x,y,t)+\frac{\partial I}{\partial x}u+\frac{\partial I}{\partial y}v+\frac{\partial I}{\partial t}$$
olur. Parlaklık sabitliğiyle birleştirince
$$I_xu+I_yv+I_t=0,\qquad \nabla I\cdot\mathbf u=-I_t.$$
Bu **optik akış denklemidir**. İki bilinmeyen ($u$, $v$) için tek denklem vardır, yani problem eksik belirlenmiştir.
:::

:::uyarı Kaynak düzeltmesi (Lucas–Kanade slayt 5 ve 7)
Slayt 5'teki Taylor açılımında $v$'nin katsayısı da $\partial I/\partial x$ yazılmış; doğrusu $\partial I/\partial y$'dir, yani $I_xu+I_yv+I_t$. Slayt 7'de $\mathbf A$'nın son satırı $[I_y^{25}\ I_y^{25}]$ olarak yazılmış; doğrusu $[I_x^{25}\ I_y^{25}]$'tir. Ayrıca $I_xu+I_yv+I_t=0$ denklemleri $\mathbf A\mathbf m=-\mathbf b$ verir, $\mathbf b=[I_t^1,\dots,I_t^{25}]^T$ ile; çözüm $\mathbf m=-(\mathbf A^T\mathbf A)^{-1}\mathbf A^T\mathbf b$'dir. Slayttaki $\mathbf A\mathbf m=\mathbf b$ biçimi ancak $\mathbf b$, $-I_t$'lerden kurulursa doğrudur; aksi hâlde kestirilen akışın işareti ters çıkar.
:::

**Açıklık problemi.** Denklem, akışın yalnız **gradyan yönündeki** bileşenini belirler. $\mathbf u$'yu gradyana paralel ve dik iki bileşene ayıralım. Dik bileşen $\nabla I$ ile çarpılınca sıfır verir, dolayısıyla denklemde görünmez. Gradyana dik yön kenar boyuncadır: kenar boyunca hareket görülemez. Ölçülebilen tek nicelik **normal akıştır**:
$$\mathbf u_\perp=-\frac{I_t}{\|\nabla I\|^2}\nabla I.$$
Hız düzleminde $(u,v)$, denklem bir doğrudur ve normal akış bu doğrunun orijine en yakın noktasıdır. İki farklı gradyan yönü (bir köşe ya da doku) iki doğru verir ve kesişimleri akışı tekleştirir. Kesin bir çözüm için bu yüzden ek varsayımlar, yani **düzenleme** gerekir. Bunlar genellikle fiziksel bir yoruma dayanır: bir şeyin korunumu, komşu piksellerin benzer hareket etmesi, gerçek nesnelerin anlık 180° dönüş yapamaması gibi.

<figure class="fig wide"><svg viewBox="0 0 640 288" role="img" aria-label="Hız düzleminde optik akış kısıtı: kenarda tek doğru ve normal akış, köşede iki doğrunun kesişimi"><text x="30" y="18" class="sm">kenar: tek kısıt (açıklık)</text><path d="M37.0,220.0 L293.5,220.0" class="mute"/><path d="M100.0,247.0 L100.0,26.5" class="mute"/><text x="297.5" y="224.0" class="sm">u</text><text x="96.0" y="22.5" class="sm" text-anchor="end">v</text><path d="M96.0,32.1 L100.2,40.5 L104.4,48.8 L108.6,57.2 L112.8,65.6 L117.0,74.0 L121.2,82.4 L125.4,90.8 L129.6,99.2 L133.8,107.6 L138.0,116.0 L142.2,124.4 L146.4,132.7 L150.6,141.1 L154.8,149.5 L159.0,157.9 L163.2,166.3 L167.3,174.7 L171.5,183.1 L175.7,191.5 L179.9,199.9 L184.1,208.3 L188.3,216.6 L192.5,225.0 L196.7,233.4 L200.9,241.8" class="ca w2" fill="none"/><path d="M100.0,220.0 L172.0,184.0" class="cp w2" fill="none"/><path d="M166.8,190.1 L172.0,184.0 L164.0,184.5" class="cp w2" fill="none"/><path d="M100.0,220.0 L145.0,130.0" class="mute w2" fill="none"/><path d="M144.5,138.0 L145.0,130.0 L138.9,135.2" class="mute w2" fill="none"/><text x="178.0" y="198.0" class="tp">normal akış (1.6, 0.8)</text><text x="139.0" y="124.0" class="sm" text-anchor="end">gerçek (1, 2)</text><text x="30" y="262" class="sm">2u + v = 4: doğru üzerindeki her (u, v)</text><text x="30" y="278" class="sm">aynı I<tspan baseline-shift="-3" font-size="75%">t</tspan> değerini verir.</text><text x="340" y="18" class="sm">köşe: iki kısıt, tek çözüm</text><path d="M347.0,220.0 L603.5,220.0" class="mute"/><path d="M410.0,247.0 L410.0,26.5" class="mute"/><text x="607.5" y="224.0" class="sm">u</text><text x="406.0" y="22.5" class="sm" text-anchor="end">v</text><path d="M406.0,32.1 L410.2,40.5 L414.4,48.8 L418.6,57.2 L422.8,65.6 L427.0,74.0 L431.2,82.4 L435.4,90.8 L439.6,99.2 L443.8,107.6 L448.0,116.0 L452.2,124.4 L456.4,132.7 L460.6,141.1 L464.8,149.5 L469.0,157.9 L473.2,166.3 L477.3,174.7 L481.5,183.1 L485.7,191.5 L489.9,199.9 L494.1,208.3 L498.3,216.6 L502.5,225.0 L506.7,233.4 L510.9,241.8" class="ca w2" fill="none"/><path d="M351.5,181.8 L355.7,179.7 L359.9,177.6 L364.1,175.5 L368.3,173.4 L372.5,171.3 L376.7,169.2 L380.9,167.1 L385.1,165.0 L389.3,162.9 L393.4,160.8 L397.6,158.7 L401.8,156.6 L406.0,154.5 L410.2,152.4 L414.4,150.3 L418.6,148.2 L422.8,146.1 L427.0,144.0 L431.2,141.9 L435.4,139.8 L439.6,137.7 L443.8,135.6 L448.0,133.5 L452.2,131.4 L456.4,129.3 L460.6,127.2 L464.8,125.1 L469.0,123.0 L473.2,120.9 L477.3,118.8 L481.5,116.7 L485.7,114.6 L489.9,112.5 L494.1,110.4 L498.3,108.3 L502.5,106.2 L506.7,104.1 L510.9,102.0 L515.1,99.9 L519.3,97.9 L523.5,95.8 L527.7,93.7 L531.9,91.6 L536.1,89.5 L540.3,87.4 L544.5,85.3 L548.7,83.2 L552.9,81.1 L557.1,79.0 L561.2,76.9 L565.4,74.8 L569.6,72.7 L573.8,70.6 L578.0,68.5 L582.2,66.4 L586.4,64.3 L590.6,62.2 L594.8,60.1 L599.0,58.0" class="cb w2" fill="none"/><circle cx="455.0" cy="130.0" r="5" class="cp-f"/><text x="463.0" y="124.0" class="tp">LK çözümü (1, 2)</text><text x="340" y="262" class="sm">2u + v = 4 ve −u + 2v = 3:</text><text x="340" y="278" class="sm">AᵀA tekil değil.</text></svg><figcaption>Şekil 1 — Bir piksel yalnız gradyan yönündeki bileşeni belirler: kenarda çözüm bir doğrudur ve en küçük normlu çözüm normal akıştır (1.6, 0.8); iki farklı gradyan yönü (köşe) çözümü tekleştirir (hesaplandı).</figcaption></figure>

### Lucas–Kanade {#lk}

Lucas ve Kanade (1981) en eski optik akış yöntemlerindendir. Optik akış, gözlemci ile sahne arasındaki göreli hareketten doğan nesne, yüzey ve kenarların görünür hareket desenidir ve çok eski bir problemdir (Horn ve Schunck 1981). Uygulamaları sıkıştırma, hareket tespiti ve nesne izlemedir. Yaklaşımlar üç ailedir: türevsel (gradyan tabanlı), korelasyon tabanlı (blok eşleme) ve öznitelik tabanlı.

Lucas–Kanade türevsel bir yöntemdir ve üç varsayıma dayanır:

1. **Parlaklık sabitliği.**
2. **Küçük hareket:** doğrusallaştırmaya izin verecek kadar.
3. **Uzamsal tutarlılık:** komşu pikseller benzer hareket eder.

Üçüncü varsayım düzenlemedir. Akış bir pikselin çevresindeki küçük bir pencerede (5 × 5, 7 × 7) sabit kabul edilir ve penceredeki her piksel bir denklem verir. 5 × 5'te iki bilinmeyen için 25 denklem olur:
$$\mathbf A=\begin{bmatrix}I_x^1&I_y^1\\\vdots&\vdots\\I_x^{25}&I_y^{25}\end{bmatrix},\qquad \mathbf A\begin{bmatrix}u\\v\end{bmatrix}=-\begin{bmatrix}I_t^1\\\vdots\\I_t^{25}\end{bmatrix}.$$

:::türetim En küçük kareler çözümü ve yapı tensörü
Aşırı belirlenmiş sistemde $E(\mathbf m)=\|\mathbf A\mathbf m+\mathbf b\|^2=\sum_i(I_x^iu+I_y^iv+I_t^i)^2$ en küçüklenir. Gradyan sıfırlanınca normal denklemler çıkar:
$$\nabla_{\mathbf m}E=2\mathbf A^T(\mathbf A\mathbf m+\mathbf b)=0\;\Rightarrow\;(\mathbf A^T\mathbf A)\,\mathbf m=-\mathbf A^T\mathbf b,$$
$$\begin{bmatrix}\sum I_x^2&\sum I_xI_y\\\sum I_xI_y&\sum I_y^2\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix}=-\begin{bmatrix}\sum I_xI_t\\\sum I_yI_t\end{bmatrix}.$$
$\mathbf A^T\mathbf A$ tanıdık bir matristir: Harris köşe dedektörünün **yapı tensörü** $M$ (M9.7), ağırlık penceresi kutu olarak. Tersinirliği özdeğerlerine bağlıdır:

- İki özdeğer de büyük (köşe, doku): çözüm kararlıdır.
- Biri büyük, biri küçük (kenar): matris neredeyse tekildir. Küçük özdeğerin özvektörü kenar boyuncadır ve o yöndeki bileşen gürültüyle belirlenir; bu, açıklık probleminin matris biçimidir.
- İkisi de küçük (düz bölge): hiçbir yön belirlenemez.

Matrisin tekil olması ancak penceredeki bütün gradyanların paralel olmasıyla mümkündür, çünkü $\mathbf A$'nın rangı 1'e düşer. Bouguet'nin uygulaması ve OpenCV bu yüzden en küçük özdeğer $\lambda_{\min}$'i pencere alanına bölüp bir eşikle karşılaştırır ve eşiğin altındaki noktaları izlemez. Bu, Shi–Tomasi'nin "izlenmeye iyi öznitelik" ölçütüdür.

El hesabı: $\sum I_x^2=10$, $\sum I_xI_y=0$, $\sum I_y^2=5$, $\sum I_xI_t=-20$, $\sum I_yI_t=5$ ise $u=20/10=2$, $v=-5/5=-1$.
:::

**Yineleme.** Doğrusallaştırma hareket büyüdükçe bozulur. Lucas ve Kanade bunu Newton–Raphson benzeri bir yinelemeyle düzeltir: bulunan $(u,v)$ ile ikinci kare geri bükülür (bilineer enterpolasyonla), kalan zamansal fark yeniden hesaplanır ve küçük bir düzeltme çözülür. $\mathbf A^T\mathbf A$ birinci karenin gradyanlarından hesaplandığı için yinelemeler arasında değişmez; yalnız sağ taraf yenilenir.

**Sınırlar.** Lucas–Kanade şu durumlarda başarısız olur:

- **büyük hareketlerde,** çünkü küçük hareket varsayımı bozulur;
- **dokusuz bölgelerde,** çünkü $\mathbf A^T\mathbf A$ kötü koşulludur;
- **parlaklık değişimlerinde,** çünkü parlaklık sabitliği bozulur.

Ders görselindeki trafik örneğinde vektörler hareketli araçlar üzerinde yoğunlaşır. Düz asfaltta akış belirlenemez, bu yüzden seyrek LK yalnız iyi köşelerde çalıştırılır.

Aşağıdaki deney üç bölgeli bir sahnede LK'yi çalıştırıyor:

- solda dokulu Gauss lekeleri;
- sağ üstte yalnız $x$'te değişen dikey çizgiler (açıklık problemi);
- sağ altta düz bir bölge.

Hareket 45° yönünde 1.5 piksellik bir ötelemedir. Varsayılan eşikle çizgili ve düz bölgelerdeki bütün noktalar reddediliyor. Eşik düşürülünce çizgili bölgenin akışı kabul ediliyor ama dikey bileşen yanlış: uç nokta hatası (EPE) 1.02 piksel. Gradyan yönündeki bileşenin hatası ise yalnız 0.054 piksel.

::lab l1

### Piramit Lucas–Kanade {#piramit}

Büyük hareketlerin çözümü bir **görüntü piramididir** (Bouguet 2001). Her seviyede görüntü yarıya küçültülür ve hareket de yarıya iner. Akış en kaba seviyede kestirilir; sonuç ikiyle çarpılıp bir alt seviyede başlangıç olarak kullanılır ve orada yalnız küçük bir artık düzeltilir. Pencere boyutu her seviyede aynıdır, yani kaba seviyede pencere görüntünün daha büyük bir kısmını kapsar. Bir seviyede güvenle bulunabilen en büyük yer değiştirme $d_{\max}$ ise $L$ seviyeli bir piramit yaklaşık $(2^{L}-1)\,d_{\max}$'a kadar hareketi yakalar; Bouguet $L=3$ ek seviye için 15 kat kazanç verir. Bedeli şudur: kaba seviyelerde ince yapılar kaybolur ve küçük, hızlı nesneler kaba seviyede hiç görünmeyebilir.

Lab l1'de 6 piksellik harekette tek seviyeli LK'nin hatası 4.47 piksel: Taylor açılımı leke boyutunun ötesinde geçersiz. 3 seviyeyle hata 0.21'e, seviye başına 5 yinelemeyle 0.11'e iniyor.

**Horn–Schunck.** Lucas–Kanade akışı yerel bir pencerede sabit kabul ederek düzenler. Horn ve Schunck (1981) ise bütün görüntüde akışın **düzgün** olmasını ister:
$$E=\iint\bigl[(I_xu+I_yv+I_t)^2+\alpha^2(\|\nabla u\|^2+\|\nabla v\|^2)\bigr]\,dx\,dy.$$
Bu enerji yoğun bir akış verir ve düz bölgelere akışı komşularından "yayar". Ama hareket sınırlarında akışı bulanıklaştırır. Modern yöntemler (TV-L1, RAFT) bu iki fikrin sağlam (robust) sürümleridir.

### Kare hızı artırma (FRUC) {#fruc}

Birçok gözetim ve izleme videosu sabit kameralarla düşük kare hızında kaydedilir. Bu ekipman ve depolama açısından ekonomiktir; uzak bir gözetim çözümünde sınırlı iletim bant genişliği de düşük kare hızını gerektirebilir. Ancak elle inceleme zorlaşır: düşük kare hızlı videoda hareket kesik kesiktir ve izleyiciyi yorar. Kare hızını artırarak daha akıcı bir video elde etmek bu yüzden önemlidir. Başka uygulamalar da vardır:

- ağır çekim;
- canlı yayında daha düşük bant genişliği;
- sınırlı hesap gücüyle video düzenleme;
- ekranlarda kare hızı uyarlama.

**Kare hızı artırma** (frame rate up-conversion, FRUC), düşük kare hızlı bir videonun var olan kareleri arasına yeni kareler üretmektir. FRUC çözülmüş (sıkıştırılmamış) kareler üzerinde çalışır. Gerçek videolardaki karmaşık, doğrusal olmayan hareketler ve aydınlatma değişimleri problemi zorlaştırır. Yöntemler:

- **Kare tekrarı:** çıkış karesinin zamandaki yerine en yakın giriş karesini kopyalamak. Yapay bozulma üretmez, ama hareket kesik kalır ve zamanla gözü yorar.
- **Kare ortalaması:** en yakın iki kareyi piksel piksel ortalamak; zamanca yakın kareye daha çok ağırlık verilir: $I_t=(1-t)I_0+tI_1$. Hareket eden nesnelerde **hayalet** üretmesi garantidir: nesne iki konumunda yarı saydam görünür.
- **Blok hareket telafili.**
- **Yoğun optik akış telafili.**
- **Derin öğrenme tabanlı** (güncel en iyi).

**Hareket telafili FRUC** (MC-FRUC) üç adımdır:

1. hareket ya da optik akış kestirimi;
2. giriş karelerini kestirilen akışa göre bükme (warping);
3. bükülmüş kareleri birleştirme.

Güncel yöntemler bu adımları evrişimli ağlarla yapar.

**Blok hareket telafili FRUC.** Bloklar arasında hareket vektörleri bulunur ve her blok ara karede telafi edilmiş konumuna kopyalanır. Önemli bir gözlem: iki kare arasındaki blok hareketi bir kayar yapboz (sliding puzzle) değildir. Blok düzeyinde hareket varsayımı tamamen geçersiz olabilir ve onarılması gereken bozulmalar beklenmelidir. Ara kare blokların ileri izdüşümüyle kurulduğunda iki sorun çıkar:

- **Delikler:** ara karenin her konumundan bir hareket vektörünün geçeceği garanti değildir.
- **Örtüşmeler:** bazı konumlara birden çok blok düşer ve bunların harmanlanması gerekir.

**Codec vektörleri.** Bütün video kodlama standartlarında kodlayıcı bir tür blok eşleme yapar. Kod çözücünün elinde düşük kaliteli de olsa blok hareket vektörleri vardır; bunlar bit akışından çıkarılıp FRUC'ta hareket kestirimini azaltmak ya da kaldırmak için kullanılabilir. Ancak kodlayıcının ayarlarına ve çalışma zamanı kararlarına göre bu vektörler gerçek hareketi betimlemeyebilir; sıkıştırma için gerçek hareket gerekmediğini yukarıda gördük. Bu vektörler doğrudan kullanılabilir ya da daha iyi bir aramanın başlangıç noktası olabilir. Her iki durumda da vektör kalitesini ölçen bir mekanizma ve kalite düşükse devreye giren bir yedek gerekir.

Aşağıdaki deney, dokulu bir arka plan üzerinde hareket eden dokulu bir kare nesneyle bu üç yöntemi karşılaştırıyor. Varsayılan harekette (6, 4) kare tekrarının PSNR'si 20.2 dB, kare ortalamasınınki 23.0 dB, blok telafili FRUC'unki 29.9 dB. BMC'de piksellerin %5.8'i delik, %5.5'i örtüşme. Üç adımlı arama aday sayısını 225'ten 25'e indiriyor, PSNR ise 27.8 dB'ye düşüyor.

::lab l2

**Optik akış telafili FRUC ve ileri–geri tutarlılık.** Yoğun akışla her piksel ayrı telafi edilir. Hareket vektörü ya da optik akış kestiricilerinin çoğunda yerleşik bir tutarlılık ve kalite denetimi vardır. Bunlardan biri **ileri–geri tutarlılık** denetimidir: kare 0'dan 1'e ileri akış $\mathbf f$ ve kare 1'den 0'a geri akış $\mathbf b$ hesaplanır; ikisi birbirinin tersi olmalıdır. Denetimi geçemeyen akış vektörleri reddedilir. Ders görselindeki siyah bölgeler bu vektörlerdir: hareketli nesnenin sınırları (örtme) ve dokusuz bölgeler. Reddedilen vektörler ya daha iyi (daha pahalı) bir yöntemle yeniden hesaplanır ya da komşu vektörlerden kestirilir. Ardından eksiksiz akış haritasıyla ara kare üretilir. Ara karede bazı **delikler** kalabilir, yani geçerli rengi olmayan pikseller; bunlar görüntü uzayında boşluk doldurma (inpainting) teknikleriyle doldurulur.

:::uyarı Kaynak düzeltmesi (lect_22 slayt 19): ileri–geri denetiminde karşılaştırma aynı pikselde yapılmaz
Slayt ileri akışın geri akışın "işareti ters çevrilmiş hâli" olması gerektiğini söylüyor. Bu ancak hareket düzgünse aynı pikselde geçerlidir. Doğru denetim, geri akışı ileri akışın gösterdiği **karşılık noktasında** okur:
$$\|\mathbf f(\mathbf x)+\mathbf b(\mathbf x+\mathbf f(\mathbf x))\|<\tau.$$
Örneğin $\mathbf f=(3,-2)$ ve karşılık noktasındaki $\mathbf b=(-2.6,2.1)$ ise fark $\|(0.4,0.1)\|=0.41$ piksel. Hareketli bir nesnenin kenarında $\mathbf b(\mathbf x)$ aynı pikselde arka planın akışını gösterebilir, bu yüzden aynı pikselde karşılaştırmak doğru vektörleri de reddeder. Sundaram ve ark. (2010) eşiği akış büyüklüğüyle ölçekler: $\|\mathbf f+\mathbf b\|^2<0.01(\|\mathbf f\|^2+\|\mathbf b\|^2)+0.5$.
:::

**RIFE.** Gerçek zamanlı ara akış kestirimi (Huang ve ark., ECCV 2022) güncel derin öğrenme yöntemlerinden biridir. Girdileri iki kare $I_0$, $I_1$ ve hedef zaman $t\in[0,1]$'dir. IFNet adlı ağ önceden eğitilmiş bir optik akış modeline dayanmadan **ara akışları** $F_{t\to0}$ ve $F_{t\to1}$ ile piksel başına bir **birleştirme haritası** $M\in[0,1]$'i uçtan uca ve önceki ağ tabanlı yöntemlerden çok daha hızlı kestirir. Zaman kodlaması girdisiyle keyfi bir $t$ desteklenir. Ara kare
$$\hat I_t=M\odot\hat I_{t\leftarrow0}+(1-M)\odot\hat I_{t\leftarrow1}$$
ile hesaplanır. $\hat I_{t\leftarrow0}$ ve $\hat I_{t\leftarrow1}$, giriş karelerinin $t$'ye **geri bükülmüş** hâlleridir; sonuç RefineNet adlı ek bir ağla iyileştirilir. Ara akışı doğrudan kestirmek, ileri izdüşümdeki delik ve örtüşme sorunlarını ortadan kaldırır: geri bükmede her çıkış pikseli tam bir kez örneklenir. $M$ ise örtülen bölgede hangi karenin güvenilir olduğunu öğrenir.

### Video stabilizasyonu {#stab}

**Motivasyon.** Sayısal video stabilizasyonu, kamera yörüngesini yumuşatarak sarsıntılı bir videoyu izlemesi rahat bir videoya dönüştürür. Kayıt cihazını kullanma koşulları zorlaştığında videoya istenmeyen salınımlar girer. Mobil cihazlarla herkesin her yerde video çekebildiği bugün bu daha da sık. İnsan elinde olmayan kameralarda da salınım olur: araç üstü otomotiv kameraları ya da insansız araç kameraları gibi. Yöntemler üç sınıftır:

- **Mekanik stabilizasyon:** jiroskopla hareketi algılayıp kamerayı hareket ettiren bir düzenekle telafi eder (gimbal). Etkilidir, ama ağırlık ve pil tüketimi gibi kaynak gereksinimleri vardır.
- **Optik stabilizasyon (OIS):** fotoğraf makinelerinde yaygındır. Kameranın açısal ve öteleme hareketlerini, görüntü sensöre düşmeden önce bir mekanizmayla telafi eder; mercek grubu ya da sensör kaydırılır. Bazı sistemler jiroskopla farklı anlardaki hız farklarını ölçüp olağan ve istenmeyen hareketleri ayırmaya çalışır.
- **Sayısal stabilizasyon:** yalnız yazılımdır (elektronik stabilizasyon, EIS). İstenmeyen kamera hareketi karelerden kestirilir ve kareler telafi edecek biçimde dönüştürülür. Maliyet avantajının yanında **daha önce kaydedilmiş** bir videoyu stabilize edebilen tek yaklaşımdır.

Sayısal stabilizasyonda iki bağlam vardır:

- **Çevrimiçi:** bir karenin dönüşümü yalnız o kare ve öncekilerle hesaplanır.
- **Çevrimdışı:** bütün kareler kullanılabilir, yani videonun tamamı önceden elde olmalıdır.

Çevrimdışı daha iyi sonuç verir, çünkü gelecekteki hareket bilinir. Çevrimiçi yöntemler ise gerçek zamanlı yayın için gereklidir.

**Problem tanımı.** Bir fotoğraf, gerçek dünyadan kesilmiş bir dikdörtgendir; bir video ise zaman boyunca uzanan bir hacimdir. Kararlı videoda bu hacim düzgün, sarsıntılı videoda kırıklıdır. Kamera hareketi genellikle altı serbestlik derecesine sahiptir: 3-B uzayda üç öteleme ve üç dönme. Sinematografideki adlarıyla öteleme hareketleri sağa/sola izleme (track), yukarı/aşağı (boom) ve ileri/geri (dolly); dönmeler kaydırma (pan), eğme (tilt) ve yuvarlanmadır (roll). 3-B kamera yolu
$$\mathbf C_{3D}=\{\mathbf H_1,\dots,\mathbf H_t\},\qquad \mathbf H_i=\begin{bmatrix}\mathbf R_i&\mathbf O_i\\\mathbf 0^T&1\end{bmatrix}$$
biçiminde yazılır. $\mathbf R_i$ 3 × 3 dönme matrisi, $\mathbf O_i$ 3 × 1 öteleme vektörüdür; $\mathbf H_i$, $i$ anında kameranın küresel koordinat sistemine göre nerede olduğunu ve nereye baktığını gösterir. $\mathbf H_1=\mathbf I$'dır, çünkü başlangıç parametrelerini temsil eder. Göreli hareket $\mathbf T_i$, $\mathbf H_i=\mathbf T_i\mathbf H_{i-1}$ ile tanımlanır.

Kararsız video $\mathbf V=\{\mathbf F_1,\dots,\mathbf F_t\}$'nin ($H\times W$ kareler) kamera yolu $\mathbf C_{3D}$, kaydedilmek istenen kararlı video $\hat{\mathbf V}$'nin düzgün yolu $\hat{\mathbf C}_{3D}$ ile bilinmeyen bir hareket gürültüsünün toplamıdır. Kesin stabilizasyon $\mathbf V\to\hat{\mathbf V}$ eşlemesi olurdu. Ancak $\hat{\mathbf C}_{3D}$ bilinmez ve $\hat{\mathbf V}$ özgün videoda hiç izdüşmemiş noktalar içerebilir. Pratikte $\hat{\mathbf V}$'ye benzeyen bir $\tilde{\mathbf V}$ üretilir. Bu yaklaşım bilgi kaybı içerir ve kare boyutu küçülebilir; $\tilde H=H$ ve $\tilde W=W$ ise stabilizasyon **tam karedir** (full-frame). Ortaya çıkan $\tilde{\mathbf C}_{3D}$ düzgün olsa da $\hat{\mathbf C}_{3D}$'den çok farklı olabilir.

**Üç adım.** Sayısal stabilizasyon üç adımdan oluşur:

1. **Kamera hareketi kestirimi:** kayıt sırasında kameranın izlediği yörünge.
2. **İstenmeyen hareketin belirlenmesi:** düzgün bir yol elde etmek için çıkarılacak hareket.
3. **Kararlı görüntünün oluşturulması:** karelerin uygun hareket telafisiyle yeniden üretilmesi.

Yöntemlerin çoğu hareket kestiriminde yalnız RGB yoğunluklarını kullanır; derinlik haritaları ya da hareket sensörleri (jiroskop) eklenince süreç kısmen sayısal olmaktan çıkar. Burada RGB tabanlı yöntemler ele alınıyor. 360° ve panoramik videolar ile birleştirme ve stabilizasyonu birlikte yapan yöntemler kapsam dışında. Yaklaşımlar klasik (her adım için ayrı algoritma) ve derin öğrenme tabanlı (denetimli ya da denetimsiz uçtan uca ağlar) olarak ikiye ayrılır.

#### Ana zorluklar {#zorluklar}

**Kamera hareketi kestirimi.** Yardımcı sensör yoksa kamera hareketi piksel yoğunluklarından çıkarılmalıdır. Hareketsiz bir sahnede bu, nesnelerin kareler arasındaki konumlarını karşılaştırarak yapılır (görünür hareket). Dinamik bir sahnede görünür hareket, kamera ve nesne hareketlerinin birleşimidir. Stabilizasyon için kamera hareketinin dış hareketlerden ayrılması gerekir. Zorluğu artıran durumlar:

- **Dokusuz arka plan.** Parlaklık sabitliği ve küçük komşuluklar varsayılır. Bir köşe tek bir eşleşme verir. Dikey bir çizgideki bir nokta çizgi boyunca birçok noktayla eşleşebilir: açıklık problemi. Dokusuz bölgede hiç çözüm yoktur ve hareket, hesaplanabildiği piksellerden çıkarılmalıdır. Çoğu yöntem baskın hareketin sabit arka plana ait olduğunu varsayar ve uzlaşı yöntemleri (RANSAC) kullanır. Ama bazen karenin en büyük bölgesi arka plan değildir. Arka plan baskın olsa bile dokusuz olduğu için izlenebilir hareketin çoğunluğunu oluşturmayabilir. Bu durumlarda PSNR ya da SSIM'i en büyükleyen iki kare hizalaması uygun olmayabilir.
- **Derinlik ve yer değiştirme belirsizliği.** Bir 2-B nokta derinliği bilinmeden 3-B'ye taşınamaz, çünkü izdüşüm doğrusu üzerinde sonsuz nokta vardır. Dinamik sahnede görüntü düzlemindeki yer değiştirme nesnenin derinliğine göre çok farklı olabilir.
- **Büyük paralaks.** Bir nesnenin iki bakış noktasından görünen konum farkıdır. Derinlikler çok farklıysa görünür hareket belirgin biçimde değişir. Ders görselinde koni aynı yerde durduğu hâlde iki bakış noktasında farklı yerlerde görünür; arka plandaki silindir ise aynı yere izdüşer.
- **Kayan deklanşör** (rolling shutter). CMOS sensörlerde satırların pozlanmaya başlama anları farklıdır.

:::uyarı Kaynak düzeltmesi (lect_23 slayt 37): kayan deklanşör satırları etkiler, sütunları değil
Slayt bozulmanın "her video **sütununun** farklı hareketi" olmasından kaynaklandığını söylüyor. CMOS sensör satır satır okunur: satır $y$'nin pozlaması ilk satırdan $y\cdot t_{\text{satır}}$ sonra başlar. Farklı hareket eden **satırlardır**. Yatay bir kaydırmada her satır biraz daha kaymış olarak kaydedilir ve dikey çizgiler **eğik** görünür; slaytın sözünü ettiği "sütunlardaki eğim" budur, ama nedeni satır zamanlamasıdır. Eğimin büyüklüğünü satır okuma süresi ve hareket hızı belirler. Kısa pozlama süresi eğimi yaratmaz; yalnız hareket bulanıklığını azaltarak eğimi keskin biçimde görünür kılar. Tek bir katı dönüşüm bu yüzden yetmez: her satıra ayrı bir dönüşüm gerekir.
:::

**İstenmeyen hareketin belirlenmesi.** Doğru kamera hareketi bilinse bile düzgün bir yol elde ederken dikkat gerektiren durumlar vardır:

- **Kasıtlı hareket.** Amaç girdideki asıl deseni bulup "hoş" ya da "iyi" bir video elde etmektir. Pratikte yöntemler "düzgün"ü tanımlayan önceden belirlenmiş kurallara dayanır, örneğin profesyonel sinematografinin iyi uygulamaları. Derin öğrenme yöntemlerinde bu kurallar açık olmayabilir; yer gerçeği videolarında ya da kayıp fonksiyonlarında örtük olarak bulunurlar. Yer gerçeği videoları büyük olasılıkla mekanik stabilizatörlü ekipmanla çekilmiştir ve ağ o ekipmanın davranışını taklit eder. Amatör bir çekimci genellikle bir plan, ekipman ya da beceriye sahip değildir ve olaylara tepki verir. Kullanıcı denetimi olmadan niyeti belirlemek zordur; özgün niyet düzgün ya da iyi uygulama da olmayabilir.
- **Yüksek ve düşük frekanslar.** El titremesinden gelen küçük, yüksek frekanslı salınımların giderilmesi gerektiği konusunda hemen herkes hemfikirdir. Ama elde kaydırma çekimleri ya da yürüyen bir kişi düşük frekanslı sallanmalar üretir. Yöntemlerin çoğu alçak geçiren süzgeçlere dayandığı için bu desenler her zaman yumuşatılmaz. Profesyonel açıdan iyi uygulama olmasa da bu hareket kasıtlı olabilir; yakınlaştırmalar için de aynısı geçerlidir. Öte yandan bir teknedeki kullanıcı için sudan gelen düşük frekanslı sallanmayı gidermek istenebilir. Ders görselindeki yumuşatılmış yol iki deseni de silip düşük frekanslı sallanmayı durgun bir kamera parçasına çevirmiştir. Şekil 3 bunu bir süzgecin frekans yanıtıyla gösteriyor.
- **Düzgünlük–sadakat ödünleşimi.** Düzgün yol için her kare $\mathbf B_{i+1}=\mathbf J_i\mathbf H_i^{-1}$ dönüşümüyle yeni bir kareye çevrilir; $\mathbf B_1=\mathbf I$. Yeni kamera görüşü özgününden ne kadar uzaksa dönüşüm o kadar güçlüdür. Görüş alanı belirgin değişir: özgün karede bilgisi olmayan bölgeler girer ve karenin önemli bir kısmı kaybolabilir. Düzgünlük ile özgün kareye benzerlik arasında bir ödünleşim gerekir. Aynı akıl yürütme optik akış tabanlı yöntemler için de geçerlidir. Bu ödünleşimin tek bir parametreyle, örneğin bir Gauss süzgecinin σ'sıyla kurulabileceği sanılabilir. Ama kamera hareketi video boyunca çok farklı olabilir ve yumuşatma sisteminin parametreleri uyarlamalı olmalıdır.
- **Uzay-zaman tutarlılığı.** 2-B yerel yörüngeler yumuşatılırken (örneğin ağlarda, meshes) uzamsal tutarlılık korunmalıdır. Yörüngeleri ayrı ayrı yumuşatmak bir karenin öznitelikleri arasında geçersiz uzamsal ilişkiler ve bozulmalar yaratır. Öznitelik yörüngeleri seyrek ve kesikli olduğu için bu hibrit yöntemlerde daha da kritiktir.
- **Parçalar arası tutarlılık.** Çok uzun yolları bir seferde yumuşatmak hesap ve özellikle bellek yüzünden olanaksız olabilir. Birçok yöntem kayan pencereler kullanır; kayan pencereler eniyileme yöntemleriyle çevrimiçi yumuşatmayı da mümkün kılar. Ayrık parçaları ayrı yumuşatanlar parçalar arasındaki süreksizliklere dikkat etmelidir. Sabit sayıda giriş ve çıkış karesi olan ağlar da kayan pencereye ihtiyaç duyar.

**Kararlı görüntünün oluşturulması.** Görüntü tabanlı oluşturma teknikleri kullanılır. Zorluklar eksik ve kaybolan bilgiyle ilgilidir ve özgün görüş ile yeni görüş arasındaki uzaklığa bağlıdır. Eksik bilgi iki biçimde ortaya çıkar: ya bölge özgün karenin görüş alanında değildir, ya da sahnedeki bir nesnenin arkasındadır. Kare kenarlarındaki eksik bilgi yalnız dönüşümün ($\mathbf B_{i+1}$) gücüne bağlıdır. Açığa çıkan bölgeler ise ayrıca sahnenin derinlik değişimine de bağlıdır. Kısacası ya kenarlarda pikseller kareye girip çıkar, ya da karenin ortasında yeni alanlar belirir. Kamera yalnız kendi eksenleri etrafında döner ve ötelenmezse yalnız kenarlarda bilgi kaybolur, çünkü 2-B dönme derinlikten bağımsız bir görünür hareket üretir. 3-B öteleme ise derinliğe bağlıdır ve hem kenarlarda hem de açığa çıkan bölgelerde eksik bilgiye yol açar.

- **Büyük bölgeleri doldurma** (inpainting). Zorluk bölgenin boyutuna ve dokunun karmaşıklığına bağlıdır; boyut, stabilizasyon için uygulanan dönüşümün gücüyle ilişkilidir. Diğer karelerden bilgiyle doldurmak bölgeleri yok saymaktan iyidir. Ama her kare farklı zaman ve konumda çekildiği için bu eşzamansız bilgi dikkatle işlenmelidir. Üretken derin öğrenme teknikleri bunu çok daha iyi yapar.
- **Tek boyutlu kırpma.** Klasik yöntemler genellikle kırpar: kareler, kararlı videoda hiçbir karede boş (bilgisiz) piksel kalmayacak biçimde kesilir. Bunun bedeli, tek bir aşırı dönüştürülmüş karenin bütün videoyu etkilemesidir.

#### Yöntemler {#stab-yontem}

**Kamera hareketi kestirimi.** Yöntemler iki gruptur:

- **Yoğunluk tabanlı:** komşu kareleri ham yoğunluklarıyla hizalar; yalnız eşleştirme adımı vardır.
- **Yerel öznitelik tabanlı:** seyrek bir öznitelik kümesi (SIFT, SURF, ORB, …) hesaplar; algılama, betimleme ve eşleştirme adımları vardır.

Eşleşen öznitelikler 2-B, 3-B ve hibrit hareket kestirim yöntemlerinin hepsinde ilk adımdır. Dokusuz arka planlarda kötü sonuç verdikleri için düzenleme gerekir. Kamera hareketini sahnedeki hareketlerden ayırmak için en çok uzlaşı yöntemleri, özellikle RANSAC kullanılır. Varsayım şudur: bulunan bütün noktalar arasındaki baskın desenin nedeni kamera hareketidir. Derinlik ve yer değiştirme belirsizliği bu ayrımı çok zorlaştırabilir.

- **2-B yöntemler** iki türdür. **Küresel** yöntemler bütün kare için tek bir dönüşüm matrisi kullanır. Bu, bütün görüntü eşit etkilendiğinde yeterlidir: derinlik değişimi azsa ya da 3-B telafi yalnız dönmeyse. Az parametreyle düşük hata verdiği için çekicidir. Büyük paralaks ve kayan deklanşör ise **yerel** hareket gerektirir. Küresel hareket homojen koordinatlarda bir dönüşüm matrisiyle temsil edilir. Düşük maliyetli ya da gerçek zamanlı uygulamalar için de yöntemler vardır:
    - 1-bit ikili görüntülerin parlaklık düzleminde XOR'la SAD hesaplayan basitleştirilmiş bir blok eşleme. Vektörler önce yerel olarak, sonra bütün kare boyunca süzülür ve küresel hareket kalanlardan hesaplanır.
    - Yalnız öteleme varsayarak satır ve sütun **integral izdüşümlerini** hizalayan yöntemler.
    - Öznitelik tabanlı yöntemler: SIFT, MSER, ORB ve RANSAC. Bunlar literatürde daha yaygındır, ama ön plan nesneleri çok büyükse RANSAC etkisiz kalır ve ek düzenleyiciler gerekir.
- **Yerel 2-B hareket** farklı ayrıntı düzeylerinde temsil edilir. Optik akış en ince düzeydir: her piksele bağımsız bir hareket. Yakın pikseller gruplanıp tek bir dönüşümle de temsil edilebilir (mesh). Yerel yöntemler genellikle önce küresel hareketi kestirir, sonra yerel olarak iyileştirir. Hepsi büyük paralaksla, bazıları kayan deklanşörle de baş eder.
- **3-B yöntemler** kamera yolunu hareketten yapı (structure from motion, SfM) teknikleriyle bulur: görüntü dizisinden 3-B yapı (bir nokta bulutu) ve kamera konumları kurulur. Öznitelik yörüngelerinden dönme ve öteleme matrisleri türetilir. İç parametre matrisi genellikle bilinmediği için iş zordur. 3-B yöntemler büyük paralaksla doğal olarak baş eder.
- **Hibrit yöntemler** ikisini birleştirir. Bazıları video boyunca 2-B öznitelik yörüngeleri kurar ama 3-B kamera konumu hesaplamaz; yörüngeleri doğrudan yumuşatır. Bunlara "2.5-B" denir ve büyük paralaksla baş ederler. Bazıları da videonun farklı parçalarında 2-B ya da 3-B hareket kullanır.

:::uyarı Kaynak düzeltmesi (lect_23 slayt 60): homografinin serbestlik derecesi 8'dir
Slayt benzerlik dönüşümünün 4, homografinin 6 serbestlik derecesi olduğunu söylüyor. Benzerlik için 4 doğrudur: öteleme 2, dönme 1, ölçek 1. **Afin** dönüşümün serbestlik derecesi 6'dır. **Homografi** 3 × 3 matrisi ölçeğe kadar tanımlıdır ve 8 serbestlik derecesine sahiptir (M9.15): 9 eleman eksi bir ölçek. Stabilizasyonda genellikle en az serbestlikli yeterli model seçilir, örneğin OpenCV'nin `estimateAffinePartial2D`'si 4 serbestlikli benzerlik dönüşümü uydurur. Modelin serbestliği arttıkça yumuşatılan yol paralaks ve yanlış eşleşmelerden kaynaklanan bozulmaları da taşıyabilir.
:::

**İstenmeyen hareketin belirlenmesi.** Hareket kestirimi nesnel bir adımdır: model ne olursa olsun bir kestirim doğru ya da yanlış diye sınıflanabilir. İstenmeyen hareketi belirleme ve süzme ise özneldir: birçok doğru ya da kabul edilebilir çıktı vardır ve süzme kullanıcının niyetiyle ilgilidir. Çoğu çalışmada bu adım kamera yolunu yumuşatmaktır (yörünge yumuşatma). Hareket kestirimi başarısızsa bu adımda düzeltilemez ve kararlı kareler doğru oluşturulamaz. Üç teknik grubu vardır:

1. **Süzme tabanlı yöntemler** iki alt türdür:
    - **Sinyal işleme yöntemleri** yüksek frekansları bastırır, en yaygını Gauss süzgecidir. Bazı çalışmalar bunun yetmediğini savunur (yüksek ve düşük frekans tartışması). Ders görselinde σ = 20 ve σ = 40 ile uyarlamalı σ karşılaştırılıyor; büyük σ ani ve kasıtlı tepeleri de düzleştiriyor.
    - **Kestirim yöntemleri** kasıtlı hareketin bir özelliğini, örneğin sabit hızı varsayar. Önemli bir örnek **Kalman süzgecidir**. Uzun süredir kullanılır, çünkü yeni kestirim için yalnız önceki durumu gerektirir ve bu yüzden çevrimiçi stabilizasyona uygundur. Sabit hız varsayımı yolu düzgünleştirir ve yumuşatmanın gücü varsayılan ivme varyansına bağlıdır: düşük değerler daha düzgün yollar, yüksek değerler özgüne daha benzer yollar üretir.
2. **Enterpolasyon tabanlı yöntemler** belirli konumlara sabit değerler atar ve aradaki değerleri polinomlarla, genellikle ikinci dereceden fonksiyonlarla enterpole eder. Düzgünlük enterpolasyon fonksiyonunun düşük derecesinden gelir. Derece ve sabit nokta sayısı arttıkça hareket özgüne benzer.
3. **Eniyileme tabanlı yöntemler** kararlı yolu çok amaçlı bir fonksiyonla bulur. Hız ve/veya ivme gibi bir kararlılık terimi en küçüklenir; yanında bilgi kaybı ve olası bozulmalar gibi kare bozulmasını azaltan terimler bulunur. Ticari olarak en başarılı stabilizasyon algoritmalarından bazıları bu yaklaşımı kullanır.

:::türetim Yörünge yumuşatma bir alçak geçiren süzgeçtir
Ölçülen yol $c_t$ ve yumuşatılmış yol $p_t$ olsun.

**Gauss süzgeci.** $p=g_\sigma*c$. Periyodu $P$ kare olan bir sinüs bileşeni
$$H(P)=\exp\!\Bigl(-\frac{2\pi^2\sigma^2}{P^2}\Bigr)$$
kazancıyla geçer; bu, Gauss'un Fourier dönüşümünün frekans $1/P$'deki değeridir. $\sigma=10$ için el titremesi ($P\approx4$) pratikte sıfırlanır ($10^{-54}$), periyodu 40 olan yürüme sallanması 0.29 kazançla kalır, uzun bir kaydırma ($P=300$) 0.98 ile geçer (Şekil 3).

**L2 eniyileme.** Bir amaç fonksiyonu da aynı biçimi verir:
$$\min_p\sum_t(p_t-c_t)^2+\lambda\sum_t(p_{t+1}-2p_t+p_{t-1})^2\;\Rightarrow\;(\mathbf I+\lambda\mathbf D^T\mathbf D)\,\mathbf p=\mathbf c.$$
Burada $\mathbf D$ ikinci fark operatörüdür. Frekans uzayında kazanç
$$\frac1{1+\lambda(2-2\cos\omega)^2}=\frac1{1+16\lambda\sin^4(\omega/2)}$$
olur, yani bu da bir alçak geçirendir. Sadakat terimi yolun özgünden uzaklaşmasını, yani kırpmayı cezalandırır.

**L1 eniyileme.** Grundmann, Kwatra ve Essa (CVPR 2011) kare normlarını L1 ile değiştirir ve birinci, ikinci ve üçüncü farkları birlikte cezalandırır. L1 seyreklik getirdiği için yol durgun, sabit hızlı ve yumuşak geçişli parçalardan oluşur; bu, profesyonel bir operatörün tripod, dolly ve kaydırma hareketlerine benzer. Ayrıca kırpma penceresinin karede kalması bir kısıt olarak eklenir. Problem doğrusal programdır.
:::

<figure class="fig wide"><svg viewBox="0 0 640 236" role="img" aria-label="Gauss yumuşatmanın kazancı periyoda göre: el titremesi, yürüme sallanması ve kaydırma bantları"><path d="M60,200.0 L390,200.0" class="mute"/><text x="54" y="204.0" text-anchor="end" class="sm">0</text><path d="M60,155.0 L390,155.0" class="mute"/><text x="54" y="159.0" text-anchor="end" class="sm">0.25</text><path d="M60,110.0 L390,110.0" class="mute"/><text x="54" y="114.0" text-anchor="end" class="sm">0.5</text><path d="M60,65.0 L390,65.0" class="mute"/><text x="54" y="69.0" text-anchor="end" class="sm">0.75</text><path d="M60,20.0 L390,20.0" class="mute"/><text x="54" y="24.0" text-anchor="end" class="sm">1</text><text x="60.0" y="216" text-anchor="middle" class="sm">2</text><text x="120.3" y="216" text-anchor="middle" class="sm">5</text><text x="166.0" y="216" text-anchor="middle" class="sm">10</text><text x="211.6" y="216" text-anchor="middle" class="sm">20</text><text x="257.3" y="216" text-anchor="middle" class="sm">40</text><text x="317.6" y="216" text-anchor="middle" class="sm">100</text><text x="390.0" y="216" text-anchor="middle" class="sm">300</text><text x="225.0" y="232" text-anchor="middle" class="sm">periyot (kare, logaritmik)</text><rect x="60.0" y="20" width="72.4" height="180" class="soft-p" opacity="0.5"/><text x="96.2" y="32" text-anchor="middle" class="sm">el titremesi</text><rect x="238.4" y="20" width="33.6" height="180" class="soft-p" opacity="0.5"/><text x="255.2" y="32" text-anchor="middle" class="sm">yürüme</text><rect x="344.3" y="20" width="45.7" height="180" class="soft-p" opacity="0.5"/><text x="367.2" y="32" text-anchor="middle" class="sm">kaydırma</text><path d="M60.0,200.0 L61.7,200.0 L63.3,200.0 L65.0,200.0 L66.6,200.0 L68.3,200.0 L69.9,200.0 L71.6,200.0 L73.3,200.0 L74.9,200.0 L76.6,200.0 L78.2,200.0 L79.9,200.0 L81.6,200.0 L83.2,200.0 L84.9,200.0 L86.5,200.0 L88.2,200.0 L89.8,200.0 L91.5,200.0 L93.2,200.0 L94.8,200.0 L96.5,200.0 L98.1,200.0 L99.8,200.0 L101.5,200.0 L103.1,200.0 L104.8,200.0 L106.4,200.0 L108.1,200.0 L109.7,200.0 L111.4,200.0 L113.1,200.0 L114.7,200.0 L116.4,199.9 L118.0,199.9 L119.7,199.9 L121.4,199.8 L123.0,199.7 L124.7,199.6 L126.3,199.5 L128.0,199.4 L129.6,199.2 L131.3,198.9 L133.0,198.6 L134.6,198.2 L136.3,197.7 L137.9,197.2 L139.6,196.6 L141.3,195.8 L142.9,195.0 L144.6,194.0 L146.2,192.9 L147.9,191.7 L149.5,190.4 L151.2,188.9 L152.9,187.2 L154.5,185.5 L156.2,183.6 L157.8,181.5 L159.5,179.3 L161.2,177.0 L162.8,174.6 L164.5,172.0 L166.1,169.3 L167.8,166.5 L169.4,163.7 L171.1,160.7 L172.8,157.6 L174.4,154.5 L176.1,151.3 L177.7,148.1 L179.4,144.8 L181.1,141.5 L182.7,138.2 L184.4,134.9 L186.0,131.6 L187.7,128.2 L189.3,124.9 L191.0,121.6 L192.7,118.3 L194.3,115.1 L196.0,111.9 L197.6,108.8 L199.3,105.7 L201.0,102.6 L202.6,99.7 L204.3,96.7 L205.9,93.9 L207.6,91.1 L209.2,88.4 L210.9,85.7 L212.6,83.1 L214.2,80.6 L215.9,78.2 L217.5,75.8 L219.2,73.6 L220.9,71.3 L222.5,69.2 L224.2,67.1 L225.8,65.1 L227.5,63.2 L229.1,61.4 L230.8,59.6 L232.5,57.8 L234.1,56.2 L235.8,54.6 L237.4,53.1 L239.1,51.6 L240.8,50.2 L242.4,48.8 L244.1,47.5 L245.7,46.3 L247.4,45.1 L249.0,43.9 L250.7,42.8 L252.4,41.8 L254.0,40.8 L255.7,39.8 L257.3,38.9 L259.0,38.0 L260.7,37.2 L262.3,36.4 L264.0,35.6 L265.6,34.9 L267.3,34.2 L268.9,33.5 L270.6,32.9 L272.3,32.3 L273.9,31.7 L275.6,31.1 L277.2,30.6 L278.9,30.1 L280.6,29.6 L282.2,29.1 L283.9,28.7 L285.5,28.3 L287.2,27.9 L288.8,27.5 L290.5,27.1 L292.2,26.8 L293.8,26.5 L295.5,26.2 L297.1,25.9 L298.8,25.6 L300.5,25.3 L302.1,25.1 L303.8,24.8 L305.4,24.6 L307.1,24.4 L308.7,24.1 L310.4,23.9 L312.1,23.7 L313.7,23.6 L315.4,23.4 L317.0,23.2 L318.7,23.1 L320.4,22.9 L322.0,22.8 L323.7,22.6 L325.3,22.5 L327.0,22.4 L328.6,22.3 L330.3,22.2 L332.0,22.1 L333.6,22.0 L335.3,21.9 L336.9,21.8 L338.6,21.7 L340.3,21.6 L341.9,21.5 L343.6,21.4 L345.2,21.4 L346.9,21.3 L348.5,21.2 L350.2,21.2 L351.9,21.1 L353.5,21.1 L355.2,21.0 L356.8,21.0 L358.5,20.9 L360.2,20.9 L361.8,20.8 L363.5,20.8 L365.1,20.8 L366.8,20.7 L368.4,20.7 L370.1,20.6 L371.8,20.6 L373.4,20.6 L375.1,20.6 L376.7,20.5 L378.4,20.5 L380.1,20.5 L381.7,20.5 L383.4,20.4 L385.0,20.4 L386.7,20.4 L388.3,20.4 L390.0,20.4" class="cp w2" fill="none"/><path d="M60.0,200.0 L61.7,200.0 L63.3,200.0 L65.0,200.0 L66.6,200.0 L68.3,200.0 L69.9,200.0 L71.6,200.0 L73.3,200.0 L74.9,200.0 L76.6,200.0 L78.2,200.0 L79.9,200.0 L81.6,200.0 L83.2,200.0 L84.9,200.0 L86.5,200.0 L88.2,200.0 L89.8,200.0 L91.5,200.0 L93.2,200.0 L94.8,200.0 L96.5,200.0 L98.1,200.0 L99.8,200.0 L101.5,200.0 L103.1,200.0 L104.8,200.0 L106.4,200.0 L108.1,200.0 L109.7,200.0 L111.4,200.0 L113.1,200.0 L114.7,200.0 L116.4,200.0 L118.0,200.0 L119.7,200.0 L121.4,200.0 L123.0,200.0 L124.7,200.0 L126.3,200.0 L128.0,200.0 L129.6,200.0 L131.3,200.0 L133.0,200.0 L134.6,200.0 L136.3,200.0 L137.9,200.0 L139.6,200.0 L141.3,200.0 L142.9,200.0 L144.6,200.0 L146.2,200.0 L147.9,200.0 L149.5,200.0 L151.2,200.0 L152.9,200.0 L154.5,200.0 L156.2,200.0 L157.8,200.0 L159.5,200.0 L161.2,200.0 L162.8,200.0 L164.5,200.0 L166.1,200.0 L167.8,200.0 L169.4,200.0 L171.1,200.0 L172.8,200.0 L174.4,200.0 L176.1,200.0 L177.7,200.0 L179.4,200.0 L181.1,200.0 L182.7,200.0 L184.4,200.0 L186.0,200.0 L187.7,200.0 L189.3,200.0 L191.0,200.0 L192.7,200.0 L194.3,200.0 L196.0,199.9 L197.6,199.9 L199.3,199.9 L201.0,199.8 L202.6,199.7 L204.3,199.6 L205.9,199.5 L207.6,199.3 L209.2,199.1 L210.9,198.8 L212.6,198.5 L214.2,198.1 L215.9,197.7 L217.5,197.1 L219.2,196.4 L220.9,195.7 L222.5,194.8 L224.2,193.8 L225.8,192.7 L227.5,191.5 L229.1,190.1 L230.8,188.6 L232.5,186.9 L234.1,185.1 L235.8,183.2 L237.4,181.1 L239.1,178.9 L240.8,176.6 L242.4,174.1 L244.1,171.5 L245.7,168.8 L247.4,166.0 L249.0,163.1 L250.7,160.1 L252.4,157.1 L254.0,153.9 L255.7,150.7 L257.3,147.5 L259.0,144.2 L260.7,140.9 L262.3,137.6 L264.0,134.3 L265.6,130.9 L267.3,127.6 L268.9,124.3 L270.6,121.0 L272.3,117.8 L273.9,114.5 L275.6,111.3 L277.2,108.2 L278.9,105.1 L280.6,102.1 L282.2,99.1 L283.9,96.2 L285.5,93.4 L287.2,90.6 L288.8,87.9 L290.5,85.2 L292.2,82.7 L293.8,80.2 L295.5,77.8 L297.1,75.4 L298.8,73.1 L300.5,70.9 L302.1,68.8 L303.8,66.8 L305.4,64.8 L307.1,62.9 L308.7,61.0 L310.4,59.2 L312.1,57.5 L313.7,55.9 L315.4,54.3 L317.0,52.8 L318.7,51.3 L320.4,49.9 L322.0,48.6 L323.7,47.3 L325.3,46.0 L327.0,44.9 L328.6,43.7 L330.3,42.6 L332.0,41.6 L333.6,40.6 L335.3,39.6 L336.9,38.7 L338.6,37.9 L340.3,37.0 L341.9,36.2 L343.6,35.5 L345.2,34.7 L346.9,34.0 L348.5,33.4 L350.2,32.7 L351.9,32.1 L353.5,31.6 L355.2,31.0 L356.8,30.5 L358.5,30.0 L360.2,29.5 L361.8,29.1 L363.5,28.6 L365.1,28.2 L366.8,27.8 L368.4,27.4 L370.1,27.1 L371.8,26.7 L373.4,26.4 L375.1,26.1 L376.7,25.8 L378.4,25.5 L380.1,25.3 L381.7,25.0 L383.4,24.8 L385.0,24.5 L386.7,24.3 L388.3,24.1 L390.0,23.9" class="ca w2" fill="none"/><path d="M60.0,200.0 L61.7,200.0 L63.3,200.0 L65.0,200.0 L66.6,200.0 L68.3,200.0 L69.9,200.0 L71.6,200.0 L73.3,200.0 L74.9,200.0 L76.6,200.0 L78.2,200.0 L79.9,200.0 L81.6,200.0 L83.2,200.0 L84.9,200.0 L86.5,200.0 L88.2,200.0 L89.8,200.0 L91.5,200.0 L93.2,200.0 L94.8,200.0 L96.5,200.0 L98.1,200.0 L99.8,200.0 L101.5,200.0 L103.1,200.0 L104.8,200.0 L106.4,200.0 L108.1,200.0 L109.7,200.0 L111.4,200.0 L113.1,200.0 L114.7,200.0 L116.4,200.0 L118.0,200.0 L119.7,200.0 L121.4,200.0 L123.0,200.0 L124.7,200.0 L126.3,200.0 L128.0,200.0 L129.6,200.0 L131.3,200.0 L133.0,200.0 L134.6,200.0 L136.3,200.0 L137.9,200.0 L139.6,200.0 L141.3,200.0 L142.9,200.0 L144.6,200.0 L146.2,200.0 L147.9,200.0 L149.5,200.0 L151.2,200.0 L152.9,200.0 L154.5,200.0 L156.2,200.0 L157.8,200.0 L159.5,200.0 L161.2,200.0 L162.8,200.0 L164.5,200.0 L166.1,200.0 L167.8,200.0 L169.4,200.0 L171.1,200.0 L172.8,200.0 L174.4,200.0 L176.1,200.0 L177.7,200.0 L179.4,200.0 L181.1,200.0 L182.7,200.0 L184.4,200.0 L186.0,200.0 L187.7,200.0 L189.3,200.0 L191.0,200.0 L192.7,200.0 L194.3,200.0 L196.0,200.0 L197.6,200.0 L199.3,200.0 L201.0,200.0 L202.6,200.0 L204.3,200.0 L205.9,200.0 L207.6,200.0 L209.2,200.0 L210.9,200.0 L212.6,200.0 L214.2,200.0 L215.9,200.0 L217.5,200.0 L219.2,200.0 L220.9,200.0 L222.5,200.0 L224.2,200.0 L225.8,200.0 L227.5,200.0 L229.1,200.0 L230.8,200.0 L232.5,200.0 L234.1,200.0 L235.8,200.0 L237.4,200.0 L239.1,200.0 L240.8,200.0 L242.4,200.0 L244.1,200.0 L245.7,200.0 L247.4,200.0 L249.0,200.0 L250.7,200.0 L252.4,200.0 L254.0,200.0 L255.7,200.0 L257.3,200.0 L259.0,200.0 L260.7,200.0 L262.3,200.0 L264.0,200.0 L265.6,200.0 L267.3,200.0 L268.9,199.9 L270.6,199.9 L272.3,199.8 L273.9,199.8 L275.6,199.7 L277.2,199.6 L278.9,199.4 L280.6,199.2 L282.2,199.0 L283.9,198.7 L285.5,198.4 L287.2,198.0 L288.8,197.5 L290.5,196.9 L292.2,196.2 L293.8,195.4 L295.5,194.5 L297.1,193.4 L298.8,192.3 L300.5,191.0 L302.1,189.6 L303.8,188.0 L305.4,186.3 L307.1,184.4 L308.7,182.5 L310.4,180.3 L312.1,178.1 L313.7,175.7 L315.4,173.2 L317.0,170.5 L318.7,167.8 L320.4,165.0 L322.0,162.0 L323.7,159.0 L325.3,155.9 L327.0,152.8 L328.6,149.6 L330.3,146.3 L332.0,143.0 L333.6,139.7 L335.3,136.4 L336.9,133.0 L338.6,129.7 L340.3,126.4 L341.9,123.1 L343.6,119.8 L345.2,116.6 L346.9,113.3 L348.5,110.2 L350.2,107.1 L351.9,104.0 L353.5,101.0 L355.2,98.0 L356.8,95.1 L358.5,92.3 L360.2,89.6 L361.8,86.9 L363.5,84.3 L365.1,81.7 L366.8,79.3 L368.4,76.9 L370.1,74.6 L371.8,72.3 L373.4,70.2 L375.1,68.1 L376.7,66.0 L378.4,64.1 L380.1,62.2 L381.7,60.4 L383.4,58.6 L385.0,56.9 L386.7,55.3 L388.3,53.7 L390.0,52.2" class="cb w2" fill="none"/><text x="410" y="34" class="sm">Gauss süzgecinin kazancı</text><text x="410" y="52" class="sm">H = exp(−2π²σ²/P²)</text><text x="410" y="80" class="tp">σ = 3: yürüme (P = 40) × 0.89</text><text x="410" y="100" class="ta">σ = 10: yürüme (P = 40) × 0.29</text><text x="410" y="120" class="tb">σ = 30: yürüme (P = 40) × 1.5·10<tspan baseline-shift="4" font-size="75%">−5</tspan></text><text x="410" y="150" class="sm">σ = 10: titreme (P = 4) × 3·10<tspan baseline-shift="4" font-size="75%">−54</tspan>,</text><text x="410" y="168" class="sm">kaydırma (P = 300) × 0.978.</text><text x="410" y="196" class="sm">Büyük σ yürümeyi de siler,</text><text x="410" y="214" class="sm">kırpma gereksinimini artırır.</text></svg><figcaption>Şekil 3 — Yörünge yumuşatma bir alçak geçiren süzgeçtir: titremeyi her σ siler, yürüme sallanmasının ne kadar kalacağını σ belirler, kasıtlı kaydırma geçer (hesaplandı).</figcaption></figure>

Lab l3 bu ödünleşimleri 300 karelik bir yörüngede ölçüyor. Yörünge durgun → kaydırma → durgun bir kasıtlı hareket, yürüme sallanması ve el titremesinden oluşuyor:

- **Çevrimdışı Gauss:** σ = 3'ten 60'a çıkınca kalan titreme 0.28'den 0.013'e iniyor, gereken kırpma 11'den 87 piksele çıkıyor.
- **Çevrimiçi Gauss** (σ = 10, yalnız geçmiş): kaydırmayı 7.8 kare geriden izliyor, kırpma 50 piksel. Çevrimdışısında kırpma 19 piksel.
- **Kalman:** ivme varyansı küçüldükçe yol düzleşiyor ama kaydırmayı geriden izliyor (q = 10⁻⁴'te 6.2 kare gecikme, 52 piksel kırpma).

::lab l3

**Oluşturma yöntemleri.** Düzgün hareketi oluşturacak görüş konumları bulunduktan sonra kararlı kareler oluşturulur. Yöntemler varsayımlarına ve geometri kullanımlarına göre sınıflanır; hepsi özgün karenin piksellerini yeni kareye yeniden izdüşürür, derinlik bilgisi genellikle örtüktür ve yerel yer değiştirmeyle ilişkilidir. Asıl sorun dönüştürülmüş görüntüdeki boşluklardır: açığa çıkan bölgeler ve görüş alanı dışında kalan kenarlar.

- **Doldurma tabanlı** yöntemler boşlukları yeni bilgiyle doldurur. Tercih edilirler ama yapay bozulma getirebilirler.
- **Kırpma tabanlı** yöntemler görüş alanını daraltır ama bozulma riski almaz. Ticari yöntemler bu yüzden ağırlıkla bunları kullanır.

Küresel hareketli 2-B yöntemlerde oluşturma, dönüşüm matrisini özgün karelere uygulamaktır: kare tekdüze dönüştürüldüğü için açığa çıkma olmaz ve klasik enterpolasyon yeter. Kırpma tabanlı yöntemler iki ölçütü izler:

- **boyut tutarlılığı:** her karenin boyutu aynıdır;
- **piksel geçerliliği:** her kare yalnız geçerli bilgili piksellerden oluşur.

Birlikte, her kenarda en çok kesilen kare bütün videonun kırpmasını belirler. Kırpma penceresinin biçimi ve boyutu kullanılan hareket modeline bağlıdır.

**Derin öğrenme ve kaynaklar.** Derin öğrenme yöntemleri hareket kestirimini, yumuşatmayı ve oluşturmayı uçtan uca öğrenir. Örneğin Google'ın Deep Online Fused Video Stabilization yöntemi (WACV 2022) jiroskop verisini görüntüyle birleştirir. Klasik ve iyi çalışan bir algoritma, Google ve Georgia Tech'in L1 en uygun kamera yollarıdır (CVPR 2011). Yöntemlerin kronolojik bir listesi `awesome-video-stabilization` deposunda bulunur.

### OpenCV ve PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| seyrek LK | `p0 = cv2.goodFeaturesToTrack(g0, 200, 0.01, 7)`; `p1, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3, minEigThreshold=1e-4)` |
| yoğun akış | `cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)`; `cv2.DISOpticalFlow_create()`; derin: `torchvision.models.optical_flow.raft_large` |
| geri bükme | `cv2.remap(I1, map_x, map_y, cv2.INTER_LINEAR)`; PyTorch: `F.grid_sample(I1, grid, align_corners=True)` |
| küresel hareket | `M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)` (4 serbestlik) |
| stabilizasyon | yörünge: `np.cumsum([dx, dy, dθ])`; yumuşatma: `scipy.ndimage.gaussian_filter1d(traj, σ, axis=0)`; düzeltme: `cv2.warpAffine` |
| FRUC (derin) | RIFE: `hzwer/ECCV2022-RIFE`; ffmpeg: `minterpolate=fps=60:mi_mode=mci` (blok hareket telafili) |

### Özet {#ozet}

- Video, uzay-zaman sinyalidir $s_c(x_1,x_2,t)$. Çok kare yöntemleri hareket duyarlı ya da hareket telafilidir ve hareket bilgisi ister.
- Zamansal ortalama gürültüyü $\sigma/\sqrt N$'e indirir (10 → 5), hareket varsa hayalet bırakır. Hareket tespiti ve telafisi bunu önler.
- Optik akış kötü konumlanmıştır: örtme, açıklık, paralaks. Hareket modelleri görüntü, piksel, blok ve nesne düzeyindedir. Sıkıştırma için gerçek hareket gerekmez, FRUC için gerekir.
- Blok eşleme: SAD/MAD/SSD; tam arama ±7'de 225 aday ve 57 600 fark, üç adımlı arama 25 aday.
- Optik akış denklemi $I_xu+I_yv+I_t=0$ yalnız normal akışı belirler. Lucas–Kanade: pencerede en küçük kareler, $\mathbf A^T\mathbf A$ yapı tensörüdür; el hesabı $(2,-1)$. Yineleme ve piramit büyük hareketi yakalar.
- FRUC yöntemleri: tekrar, ortalama (hayalet), blok telafili (delik ve örtüşme), akış telafili (ileri–geri denetimi $\|\mathbf f+\mathbf b(\mathbf x+\mathbf f)\|$), RIFE.
- Stabilizasyon üç adımdır: kamera hareketini kestir, istenmeyen hareketi belirle, kararlı kareleri oluştur. Gauss, Kalman ve L1/L2 eniyileme birer alçak geçiren süzgeçtir. Düzgünlük ile kırpma arasında ödünleşim vardır, çevrimiçi yöntemler gecikir.
