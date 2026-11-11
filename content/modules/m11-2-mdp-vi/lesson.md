### Markov zincirinden karar sürecine {#mdp}

M11.1 durum değişimlerini eylemleri ve ödülleri yok sayarak bir Markov zinciriyle modelledi. **Markov karar süreci** (MDP), RL'nin temelidir ve zinciri iki ekle genişletir. Her adımda ajan bir **eylem** $a_t$ seçer ve sonraki durumun dağılımı bu eyleme bağlı olur. Her adımda bir **ödül** $r_t$ alınır. Zaman çizelgesi $s_0\xrightarrow{a_0,\ r_0}s_1\xrightarrow{a_1,\ r_1}s_2\cdots$ biçimine döner. Bu modül MDP'yi tanımlar, sonlu ufukta en iyi politikayı geriye doğru dinamik programlamayla bulur ve sonsuz ufukta değer yinelemesinin neden ve ne hızla yakınsadığını kanıtlar.

**Bileşenler.** Bir MDP şu altılıdır:

- Durum kümesi $\mathcal S=\{S_0,\dots,S_M\}$ ve eylem kümesi $\mathcal A=\{A_0,\dots,A_N\}$.
- Geçiş modeli $P(s_{t+1}\mid s_t,a_t)$: her durum–eylem çifti için sonraki durumun dağılımı.
- Ödül fonksiyonu $R(s_t,a_t)=r_t$: durum–eylem çiftlerini sayılara eşler. Burada belirlenimci ve zamandan bağımsız kabul edilir. Genelde ödül de rastgele olabilir; o zaman beklentisi kullanılır ve dağılımının zamanla değişmediği varsayılır. Son adımın ödülü çoğu zaman farklıdır, ör. kazanma ya da kaybetme.
- İndirim çarpanı $\gamma\in[0,1]$ ve ufuk $h$ (adım sayısı).

**Politika** $\pi$, gözlenen durumdan eylem seçen kuraldır: $a_t=\pi(s_t)$. Durumlar tam gözlenebilir olduğu için eylemi yalnız şimdiki duruma göre seçmek yeterlidir. Daha karmaşık kurallar da düşünülebilir: o anki zamanı ya da o ana kadar biriken ödülü hesaba katmak gibi. Aşağıda sonlu ufukta en iyi politikanın gerçekten zamana bağlı olduğu görülecek. **Amaç**, ödüllerin (indirimli) toplamını en çoklayan politikayı ya da en iyi politikalardan birini bulmaktır.

**Varsayımlar.** Bu modül ve M11.3 **planlama** problemini ele alır. Durumlar tam ve gürültüsüz gözlenir. Zaman, durum ve eylemler kesiklidir. Model, yani $P$ ve $R$, tümüyle bilinir. Ajanın ortamı deneyerek öğrenmesi gerekmez. Modelin bilinmediği durumu M11.4'ten itibaren modelsiz yöntemler çözer.

:::uyarı Kaynak düzeltmesi (lect_3): "eylemler belirlenimci" varsayımı
Notlar varsayımlar arasında "eylemler bilinir ve belirlenimcidir" diyor. Hemen ardından gelen stok örneğinde ise aynı eylem (yüksek fiyat) %30 olasılıkla satış, %70 olasılıkla satışsızlıkla sonuçlanıyor: geçişler rastgeledir. Planlamanın gerçek varsayımı belirlenimcilik değil, **modelin bilinmesidir**. Geçiş olasılıkları $P(s'\mid s,a)$ ve ödül fonksiyonu tam bilinir, ama bir eylemin sonucu yine rastgele olabilir. Değer yinelemesindeki $\sum_{s'}P(s'\mid s,a)V(s')$ beklentisi tam da bu rastgeleliği hesaba katar.
:::

### Örnek: robot süpürge {#supurge}

Bir robot süpürge iki odayı (salon, mutfak) pil tasarrufu yaparak temizlemeye çalışıyor. Her oda temiz ya da kirli, pil yüksek ya da düşük olabilir. Eylemler şunlardır: salonu temizle, mutfağı temizle, öbür odaya geç, şarj ol, bekle. Ödüller şöyle verilir: bir odayı başarıyla temizlemek +10, pil harcayan her temizlik −1, iki oda da temizse +20, pilin tamamen bitmesi −50. Akla yatkın bir politika şuna benzer: iki oda kirli ve pil yüksekse en yakın odayı temizle; bir oda kirli ve pil düşükse şarj ol; her yer temizse bekle.

Durum tanımı modellemenin en önemli adımıdır. Kaynaktaki listede durum (salon, mutfak, pil) üçlüsüdür: $2\cdot2\cdot2=8$ durum. Ama eylemler arasında "öbür odaya geç" ve politikada "en yakın odayı temizle" vardır. Bu kararlar robotun **hangi odada olduğunu** bilmeden verilemez. Konum duruma eklenmezse süreç Markov olmaz, çünkü aynı gözlenen durumdan aynı eylemin sonucu robotun nerede olduğuna bağlıdır. Doğru durum uzayı $2\cdot2\cdot2\cdot2=16$ durumdur. "Pilin tamamen bitmesi" de iki düzeyli pil göstergesiyle ifade edilemez; ya üçüncü bir pil düzeyi ya da soğurucu bir "pil bitti" durumu gerekir. M11.1'deki ilke burada da geçerli: Markov özelliği durum tanımının özelliğidir.

### İndirim {#indirim}

Ufuk, oyunun ya da sürecin adım sayısıdır ve genellikle büyük ya da sonsuz kabul edilir. Sonsuz ufukta ödüllerin toplamı sınırsız büyüyebilir. Ayrıca gerçek uygulamalarda erken ödül geç ödüle tercih edilir. **İndirimli getiri** iki sorunu birden çözer:
$$G=\sum_{t=0}^{h}\gamma^tR(s_t,a_t).$$

:::türetim İndirimli toplam ne zaman sonludur?
Ödüller sınırlı olsun: $|R|\le R_{\max}$. O zaman
$$\Bigl|\sum_{t=0}^\infty\gamma^tR_t\Bigr|\le R_{\max}\sum_{t=0}^\infty\gamma^t=\frac{R_{\max}}{1-\gamma}$$
olur. Bu yalnız $\gamma<1$ için geçerlidir. Kaynak $0\le\gamma\le1$ aralığını verip "$\sum_t\gamma^tR$ sonludur" diyor. $\gamma=1$ ve sonsuz ufukta bu yanlıştır: her adımda −1 alan bir ajanın toplamı $-\infty$'dur. $\gamma=1$ ancak sonlu ufukta ya da her politikanın olasılık 1 ile bir son duruma ulaştığı ("uygun politika") problemlerde kullanılabilir. $\gamma$'nın bir yorumu da etkin ufuktur: $\sum_t\gamma^t=1/(1-\gamma)$, yani $\gamma=0.99$ yaklaşık 100 adımlık bir ufka karşılık gelir. Başka bir yorum, her adımda $1-\gamma$ olasılıkla sürecin bitmesidir.
:::

### Politika değerlendirme ve değer fonksiyonu {#deger}

Sabit bir $\pi$ politikasıyla $s_0$'dan ufkun sonuna kadar gidildiğinde beklenen kazanç, politikanın **beklenen faydasıdır**:
$$V^\pi(s_0)=\sum_{t=0}^h\gamma^t\sum_{s_t}P(s_t\mid s_0,\pi)\,R(s_t,\pi(s_t)).$$
$P(s_t\mid s_0,\pi)$, $\pi$ izlenirken $t$ adım sonra $s_t$'de olma olasılığıdır. Bu, M11.1'deki $\mathbf p_t=T_\pi^t\mathbf p_0$'ın bileşenidir. Kaynakta bu terim $P(s_t\mid s_{t-1},\pi)$ diye, yani tek adımlık geçiş olarak yazılmış. Toplam $s_t$ üzerinden alındığı ve $s_{t-1}$ serbest kaldığı için doğru terim $s_0$'dan $t$ adımlık olasılıktır. **En iyi politika** $\pi^*$, her durumda en yüksek beklenen faydayı verendir: $V^{\pi^*}(s_0)\ge V^\pi(s_0)$, her $\pi$ için. Sonlu MDP'lerde aynı anda **bütün** durumlar için en iyi olan bir politika her zaman vardır.

**Değer fonksiyonu.** Ufuk $h$ sonlu olsun. $t<h$ anında $s_t$ durumunun değeri, oradan itibaren en iyi davranılırsa elde edilecek kazançtır:
$$V(s_t)=\max_{a_t}\Bigl[R(s_t,a_t)+\gamma\sum_{s_{t+1}}P(s_{t+1}\mid s_t,a_t)\,V(s_{t+1})\Bigr].$$
Tanımın tamamlanması için son adımda $V(s_h)=\max_{a_h}R(s_h,a_h)$ verilir; sonraki durum olmadığı için ikinci terim yoktur. Kaynakta beklenen fayda ile değer fonksiyonu aynı harfle ($V$) gösteriliyor. Fark şudur: $V^\pi$ sabit bir politikanın değeridir, $V$ (ya da $V^*$) en iyi eylemlerin değeridir.

### Değer yinelemesi: geriye doğru dinamik programlama {#vi}

Değer fonksiyonu kendi cinsinden tanımlandığı için sondan başa doğru hesaplanır:

- **Son adım**, $t=h$: $V(s_h)=\max_aR(s_h,a)$.
- **Bir adım kala**, $t=h-1$: $V(s_{h-1})=\max_a\bigl[R(s_{h-1},a)+\gamma\sum_{s_h}P(s_h\mid s_{h-1},a)V(s_h)\bigr]$.
- **İki adım kala**, $t=h-2$: aynı formül $V(s_{h-1})$ ile.

Her adımda en iyi eylem de kaydedilir:
$$a_t^*=\arg\max_{a_t}\Bigl[R(s_t,a_t)+\gamma\sum_{s_{t+1}}P(s_{t+1}\mid s_t,a_t)V(s_{t+1})\Bigr].$$
Bu denklem **Bellman denklemidir**. Dinamik programlamanın özü, $h$ adımlık bir karar dizisini $h$ tane tek adımlık karara bölmektir. Bütün eylem dizilerini denemek $|\mathcal A|^h$ yol ister. Geriye doğru yineleme her adımda $|\mathcal S|\cdot|\mathcal A|\cdot|\mathcal S|$ işlem yapar.

Kodda indeks çoğu zaman "kalan adım sayısı"na çevrilir. $V_0^*(s)=\max_aR(s,a)$ ile başlanır ve her $t=1,\dots,h$ için
$$V_t^*(s)=\max_a\Bigl[R(s,a)+\gamma\sum_{s^+}P(s^+\mid s,a)\,V_{t-1}^*(s^+)\Bigr]$$
hesaplanır. Burada $t$, oyunun sonuna kalan adım sayısıdır (ilk biçimdeki $h-t$), $s^+$ ise bir sonraki durumdur. Algoritma hiçbir öngörü olmadan yalnız anlık ödülle başlar, her yinelemede öngörü ufkunu bir adım uzatır. En iyi politika her kalan-adım sayısı için ayrı çıkarılır: $\pi_t^*(s)=\arg\max_a[R(s,a)+\gamma\sum_{s^+}P(s^+\mid s,a)V_{t-1}^*(s^+)]$.

### Örnek: perakende stok ve dinamik fiyat {#stok}

Bir mağaza moda bir cihazın stokunu $h=5$ günlük bir kampanya sezonunda yönetiyor. Her gün fiyat stratejisine karar verir. Fazla stok depolama ücreti doğurur. Sezon sonunda kalan stok değerini yitirir ve zararına tasfiye edilir. Mağaza her gün açıktır, yani terminal durum yoktur.

- **Durumlar** stok düzeyidir: $S_0$ (stok yok), $S_1$ (az), $S_2$ (orta), $S_3$ (çok).
- **Günlük ödül** ($t<h$) yalnız duruma bağlıdır. $S_0$'da −5 (kaçan satışlar), $S_1$ ve $S_2$'de −1 (düşük depolama ücreti), $S_3$'te −2 (yüksek depolama ücreti) alınır.
- **Sezon sonu** ($t=h$) tasfiye cezası: $S_0$ 0, $S_1$ −4, $S_2$ −8, $S_3$ −12.
- **Eylemler:** yüksek fiyat ($A_1$) %30 olasılıkla satış getirir; indirim ($A_2$) %80 olasılıkla satış getirir. Satış stoku bir düzey azaltır, satışsızlıkta stok aynı kalır. $S_0$'dan, eylemden bağımsız olarak gece yeniden stoklanıp $S_1$'e geçilir.

$\gamma=1$ ile hesap sondan başlar. $V_5=[0,-4,-8,-12]$. $t=4$'te $S_1$ için:
$$Q(S_1,\text{yüksek})=-1+0.3\cdot0+0.7\cdot(-4)=-3.8,\qquad Q(S_1,\text{indirim})=-1+0.8\cdot0+0.2\cdot(-4)=-1.8.$$
İndirim seçilir, $V_4(S_1)=-1.8$. $S_0$'da eylem zorunludur: $V_4(S_0)=-5+V_5(S_1)=-9$. $S_2$ ve $S_3$'te de indirim kazanır: −5.8 ve −10.8. $t=3$'te $S_1$ için yüksek fiyat $-1+0.3(-9)+0.7(-1.8)=-4.96$, indirim $-1+0.8(-9)+0.2(-1.8)=-8.56$ verir; bu kez yüksek fiyat kazanır. Tam tablo Şekil 1'dedir.

<figure class="fig wide"><svg viewBox="0 0 640 226" role="img" aria-label="Perakende stok örneğinde beş günlük ufukta her gün ve durum için değer ve en iyi eylem tablosu"><text x="168.0" y="24" text-anchor="middle" class="sm">t = 1</text><text x="264.0" y="24" text-anchor="middle" class="sm">t = 2</text><text x="360.0" y="24" text-anchor="middle" class="sm">t = 3</text><text x="456.0" y="24" text-anchor="middle" class="sm">t = 4</text><text x="552.0" y="24" text-anchor="middle" class="sm">t = 5 (son)</text><text x="112" y="59" text-anchor="end" class="sm">S0 stok yok</text><rect x="122" y="36" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="168.0" y="52" text-anchor="middle" class="sm">-11.51</text><text x="168.0" y="66" text-anchor="middle" class="sm" font-size="10">zorunlu</text><rect x="218" y="36" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="264.0" y="52" text-anchor="middle" class="sm">-9.96</text><text x="264.0" y="66" text-anchor="middle" class="sm" font-size="10">zorunlu</text><rect x="314" y="36" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="360.0" y="52" text-anchor="middle" class="sm">-6.8</text><text x="360.0" y="66" text-anchor="middle" class="sm" font-size="10">zorunlu</text><rect x="410" y="36" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="456.0" y="52" text-anchor="middle" class="sm">-9</text><text x="456.0" y="66" text-anchor="middle" class="sm" font-size="10">zorunlu</text><rect x="506" y="36" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="552.0" y="52" text-anchor="middle" class="sm">0</text><text x="552.0" y="66" text-anchor="middle" class="sm" font-size="10">—</text><text x="112" y="99" text-anchor="end" class="sm">S1 az</text><rect x="122" y="76" width="92" height="36" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="168.0" y="92" text-anchor="middle" class="sm">-8.546</text><text x="168.0" y="106" text-anchor="middle" class="sm" font-size="10">yüksek</text><rect x="218" y="76" width="92" height="36" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="264.0" y="92" text-anchor="middle" class="sm">-6.512</text><text x="264.0" y="106" text-anchor="middle" class="sm" font-size="10">yüksek</text><rect x="314" y="76" width="92" height="36" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="360.0" y="92" text-anchor="middle" class="sm">-4.96</text><text x="360.0" y="106" text-anchor="middle" class="sm" font-size="10">yüksek</text><rect x="410" y="76" width="92" height="36" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="456.0" y="92" text-anchor="middle" class="sm">-1.8</text><text x="456.0" y="106" text-anchor="middle" class="sm" font-size="10">indirim</text><rect x="506" y="76" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="552.0" y="92" text-anchor="middle" class="sm">-4</text><text x="552.0" y="106" text-anchor="middle" class="sm" font-size="10">—</text><text x="112" y="139" text-anchor="end" class="sm">S2 orta</text><rect x="122" y="116" width="92" height="36" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="168.0" y="132" text-anchor="middle" class="sm">-6.459</text><text x="168.0" y="146" text-anchor="middle" class="sm" font-size="10">yüksek</text><rect x="218" y="116" width="92" height="36" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="264.0" y="132" text-anchor="middle" class="sm">-5.008</text><text x="264.0" y="146" text-anchor="middle" class="sm" font-size="10">yüksek</text><rect x="314" y="116" width="92" height="36" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="360.0" y="132" text-anchor="middle" class="sm">-3.6</text><text x="360.0" y="146" text-anchor="middle" class="sm" font-size="10">indirim</text><rect x="410" y="116" width="92" height="36" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="456.0" y="132" text-anchor="middle" class="sm">-5.8</text><text x="456.0" y="146" text-anchor="middle" class="sm" font-size="10">indirim</text><rect x="506" y="116" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="552.0" y="132" text-anchor="middle" class="sm">-8</text><text x="552.0" y="146" text-anchor="middle" class="sm" font-size="10">—</text><text x="112" y="179" text-anchor="end" class="sm">S3 çok</text><rect x="122" y="156" width="92" height="36" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="168.0" y="172" text-anchor="middle" class="sm">-7.334</text><text x="168.0" y="186" text-anchor="middle" class="sm" font-size="10">indirim</text><rect x="218" y="156" width="92" height="36" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="264.0" y="172" text-anchor="middle" class="sm">-6.64</text><text x="264.0" y="186" text-anchor="middle" class="sm" font-size="10">indirim</text><rect x="314" y="156" width="92" height="36" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="360.0" y="172" text-anchor="middle" class="sm">-8.8</text><text x="360.0" y="186" text-anchor="middle" class="sm" font-size="10">indirim</text><rect x="410" y="156" width="92" height="36" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="456.0" y="172" text-anchor="middle" class="sm">-10.8</text><text x="456.0" y="186" text-anchor="middle" class="sm" font-size="10">indirim</text><rect x="506" y="156" width="92" height="36" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="552.0" y="172" text-anchor="middle" class="sm">-12</text><text x="552.0" y="186" text-anchor="middle" class="sm" font-size="10">—</text><rect x="120" y="202" width="12" height="10" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="138" y="212" class="sm">yüksek fiyat en iyi</text><rect x="290" y="202" width="12" height="10" class="soft-b" stroke="currentColor" stroke-width="0.5"/><text x="308" y="212" class="sm">indirim en iyi</text><text x="440" y="212" class="sm">← hesap sondan başa doğru</text></svg><figcaption>Şekil 1 — Perakende stok MDP'si, h = 5, γ = 1. Hücrelerde V<sub>t</sub>(s) ve en iyi eylem. Aynı durumda en iyi eylem güne göre değişir: S1 son eylem gününde (t = 4) indirime, daha önce yüksek fiyata geçer (değerler betikle hesaplandı ve notlardaki tabloyla aynı).</figcaption></figure>

:::uyarı Kaynak düzeltmesi (lect_3): ara terimlerdeki yazım hataları
Notlardaki değer tablosunun bütün sayıları doğrudur. Bu modülde betikle yeniden hesaplandı ve aynı çıktı. Ama $t=2$ ve $t=1$ satırlarındaki açık hesaplarda yanlış değerler yazılmış:

- $t=2$, $S_2$ satırında $V_3(S_2)$ "−3.96" diye yazılmış; doğrusu −3.6'dır. Sonuçlar (−5.008, −5.688) doğru değerle hesaplanmıştır.
- $t=1$ satırlarında $V_2$ değerleri "−0.992, −0.792, −1.76" diye geçiyor. Doğruları sırasıyla −6.512, −5.008 ve −6.64'tür; sonuçlar yine doğru değerlerle tutar.
- $t=1$, $S_0$: "$-5+(6.512)$" yazılmış; doğrusu $-5+(-6.512)=-11.512$'dir.

Ek olarak zaman bağımlılığı slaytı "2 ya da daha çok adım kala, $t<3$" diyor. $S_1$ için yüksek fiyat $t=3$'te de en iyidir, doğrusu $t\le3$'tür.
:::

**Zaman bağımlılığı.** $S_1$'in (az stok) politikası zamana göre değişir. Son eylem gününde ($t=4$) indirim en iyidir: yaklaşan −4'lük tasfiye cezasından kaçınmak acildir. İki ya da daha fazla gün kala ($t\le3$) yüksek fiyat en iyidir. $S_1$'in depolama maliyeti (−1) stoksuz kalmanın maliyetinin (−5) yanında küçüktür. Bu yüzden sezonun başında mağaza stokunun hızla erimesini istemez ve düşük olasılıklı satış eylemini seçer. $S_2$'nin sezon sonu cezası daha ağırdır ve indirime $S_1$'den bir gün önce ($t=3$'te) geçer. $S_3$'te her gün indirim en iyidir.

**Sonlu ve sonsuz ufuk.** Sonlu ufukta en iyi politika **durağan değildir**: aynı durumda en iyi eylem kalan süreye göre değişir. Sonsuz ufukta her adımda kalan süre aynıdır (sonsuz), dolayısıyla en iyi eylem de her adımda aynıdır: politika **durağandır**. Aşağıdaki deneyde ufuk 20 güne çıkarılınca ilk günlerin politikası sabitlenir: $S_1$ ve $S_2$'de yüksek fiyat, $S_3$'te indirim. Bu, aynı MDP'nin sonsuz ufuklu en iyi politikasıdır (Şekil 2). Değişim yalnız sezonun son günlerinde kalır.

::lab l2

### Matris biçimi {#matris}

Hesapları kısaltmak için her eylem $a$ için iki nesne tanımlanır. $R_a$, $|\mathcal S|\times1$ ödül vektörüdür. $T^a$, $|\mathcal S|\times|\mathcal S|$ geçiş matrisidir ve öğesi $T^a_{ss^+}=P(s^+\mid s,a)$ olarak tanımlanır. Değer yinelemesi şu biçimi alır:
$$V_0^*=\max_aR_a,\qquad V_t^*=\max_a\bigl[R_a+\gamma T^aV_{t-1}^*\bigr].$$
Maksimum öğe öğe alınır. Buradaki $T^a$, M11.1'deki sütun-stokastik matrisin **devriğidir**: satır $s$, $s$'den çıkış olasılıklarıdır ve satır toplamları 1'dir. Değer yinelemesindeki toplam varış durumları $s^+$ üzerinden alındığı için bu yön gerekir. $(T^aV)(s)=\sum_{s^+}P(s^+\mid s,a)V(s^+)$, yani bir sonraki adımın beklenen değeridir. (Slayttaki $T^a$'nın üçüncü satırı "$p_{20}\ p_{21}\ P_{22}\ P_{32}$" diye yazılmış; doğrusu $p_{20}\ p_{21}\ p_{22}\ p_{23}$'tür.)

Ufuk $h\to\infty$ giderken yineleme yakınsıyorsa (aşağıda kanıtlanıyor) bir sabit noktaya varılır: $V_{h-1}^*\to V_\infty^*$ ve $V_h^*\to V_\infty^*$. Bellman denklemi şu sabit nokta denklemine dönüşür:
$$V_\infty^*=\max_a\bigl[R_a+\gamma T^aV_\infty^*\bigr],\qquad\pi_\infty^*=\arg\max_a\bigl[R_a+\gamma T^aV_\infty^*\bigr].$$
Bu, politika eniyilemenin bir yoludur: önce değerleri yakınsat, sonra politikayı çıkar.

**Politika değerlendirme.** Sabit bir $\pi$ için yineleme $\max$ olmadan yazılır ve politikanın ulaşacağı değerler hesaplanır. Sabit noktada
$$V^\pi=R_\pi+\gamma T^\pi V^\pi$$
sağlanır. Bu doğrusal bir denklem sistemidir ve
$$V^\pi=(I-\gamma T^\pi)^{-1}R_\pi$$
ile çözülür. Kaynak, ters matris yoksa yinelemeye ($V_t=R_\pi+\gamma T^\pi V_{t-1}$) dönülebileceğini söylüyor. $\gamma<1$ için bu durum hiç oluşmaz. $T^\pi$ stokastik olduğu için bütün özdeğerlerinin modülü en çok 1'dir. $\gamma T^\pi$'nin özdeğerleri $\gamma$'dan küçük modüllüdür ve $I-\gamma T^\pi$'nin hiçbir özdeğeri sıfır olamaz. Ters matris $\sum_k\gamma^k(T^\pi)^k$ serisiyle de yazılabilir. Tekillik yalnız $\gamma=1$'de mümkündür. Büyük $|\mathcal S|$'de yine de yineleme seçilir: ters alma $O(|\mathcal S|^3)$, bir yineleme $O(|\mathcal S|^2)$'dir (seyrek $T$'de daha az).

### Yakınsama: büzülme {#buzulme}

∞-normu bir vektörün en büyük mutlak öğesidir: $\|x\|_\infty=\max_i|x_i|$. Matrislerde en büyük mutlak satır toplamıdır. Satır-stokastik bir matris için $\|T\|_\infty=1$'dir.

:::türetim Lemma 1: politika değerlendirme operatörü bir büzülmedir
$H(V)=R+\gamma TV$ olsun. Herhangi iki $V,\tilde V$ için
$$\|H(\tilde V)-H(V)\|_\infty=\|\gamma T(\tilde V-V)\|_\infty\le\gamma\|T\|_\infty\|\tilde V-V\|_\infty=\gamma\|\tilde V-V\|_\infty.$$
Eşitsizlik, alt-çarpımsallıktan ($\|AB\|_\infty\le\|A\|_\infty\|B\|_\infty$) gelir. Operatör ancak $\gamma<1$ ise **büzülmedir**, yani uzaklıkları kesin olarak küçültür. Kaynak koşulu "$\gamma\le1$" diye yazıyor. $\gamma=1$'de eşitsizlik yalnız uzaklığın büyümediğini söyler ve yakınsama güvencesi vermez.
:::

**Teorem 1.** $\gamma<1$ ise her başlangıç $V$ için $\lim_{n\to\infty}H^{(n)}(V)=V^\pi$'dir; $H^{(n)}$, $H$'nin $n$ kez uygulanmasıdır. Kanıt: Lemma 1 $n$ kez uygulanınca $\|H^{(n)}\tilde V-H^{(n)}V\|_\infty\le\gamma^n\|\tilde V-V\|_\infty\to0$ olur. Yani bütün başlangıçlar aynı limite gider. $V=V^\pi$ seçilirse $H(V^\pi)=V^\pi$ olduğundan bu limit $V^\pi$'dir. (Banach sabit nokta teoremi ayrıca limitin var ve tek olduğunu güvence eder.) Yinelemenin $V_0^*=\max_aR_a$ ile başlaması $H^*(0)$ demektir, yani sıfır vektöründen başlayıp bir adım atmaktır.

:::türetim Teorem 2: sonlu yineleme sınırı
Pratikte sonsuz yineleme yapılamaz. $n$ adımdan sonra son değişim $\epsilon=\|H^{(n)}V-H^{(n-1)}V\|_\infty$ gözlenmiş olsun. Gerçek değere uzaklık, kalan adımların toplamıdır:
$$\|V^\pi-H^{(n)}V\|_\infty=\Bigl\|\sum_{k=1}^\infty\bigl(H^{(n+k)}V-H^{(n+k-1)}V\bigr)\Bigr\|_\infty\le\sum_{k=1}^\infty\gamma^k\epsilon=\frac{\gamma\,\epsilon}{1-\gamma}.$$
Her terim bir öncekinin en çok $\gamma$ katıdır; bu yüzden $k$. terim $\gamma^k\epsilon$ ile sınırlıdır. Toplam $k=1$'den başladığı için $\sum_{k\ge1}\gamma^k=\gamma/(1-\gamma)$'dır.
:::

:::uyarı Kaynak düzeltmesi (lect_4): Teorem 2'nin son adımı
Notlar aynı türetimin son satırında $\epsilon\sum_{t=1}^\infty\gamma^t=\epsilon/(1-\gamma)$ yazıyor ve teoremi $\|V^\pi-H^{(n)}V\|_\infty\le\epsilon/(1-\gamma)$ olarak veriyor. Toplam $t=1$'den başladığı için doğru değer $\gamma\epsilon/(1-\gamma)$'dır. Notlardaki sınır geçerlidir ama $1/\gamma$ kat gevşektir. Aşağıdaki deneyde gerçek hata sıkı sınıra neredeyse eşittir. $\gamma=0.9$ ve $\epsilon=0.01$ için sıkı sınır 0.09, notlardaki 0.1'dir.
:::

**Değer yinelemesi.** Politika değerlendirmeden farkı, sabit nokta denkleminin $\max$ yüzünden **doğrusal olmamasıdır**. $V_\infty^*$ matris tersiyle bulunamaz, ama yinelemeyle bulunabilir. Bunun için en iyilik operatörünün de bir büzülme olduğu gösterilmelidir.

:::türetim Lemma 2: Bellman en iyilik operatörü bir büzülmedir
$H^*(V)=\max_a[R_a+\gamma T^aV]$ olsun. Bir $s$ durumu sabitlensin ve genelliği bozmadan $H^*\tilde V(s)\ge H^*V(s)$ kabul edilsin. $\tilde a$, $\tilde V$ için en iyi eylem olsun. $V$ için en iyi eylem $a^*_s$, $\tilde a$'dan kötü olamaz, dolayısıyla $H^*V(s)\ge R(s,\tilde a)+\gamma\sum_{s^+}P(s^+\mid s,\tilde a)V(s^+)$ olur. Buradan
$$0\le H^*\tilde V(s)-H^*V(s)\le\gamma\sum_{s^+}P(s^+\mid s,\tilde a)\bigl[\tilde V(s^+)-V(s^+)\bigr]\le\gamma\|\tilde V-V\|_\infty$$
çıkar. Son adımda olasılıklar toplamının 1 olduğu kullanılır: bir ağırlıklı ortalama en büyük öğeyi aşamaz. $H^*V(s)\ge H^*\tilde V(s)$ durumunda roller değiştirilir. Her $s$ için $|H^*\tilde V(s)-H^*V(s)|\le\gamma\|\tilde V-V\|_\infty$ olduğuna göre ∞-normda da öyledir. Daha kısa bir yol $|\max_af(a)-\max_ag(a)|\le\max_a|f(a)-g(a)|$ eşitsizliğidir.
:::

<figure class="fig wide"><svg viewBox="0 0 640 266" role="img" aria-label="Bellman optimallik operatörünün büzülmesi: iki değer vektörü arasındaki uzaklığın yinelemeyle gamma üssü n hızında küçülmesi"><path d="M60,224.0 L420,224.0" class="mute"/><text x="54" y="228.0" text-anchor="end" class="sm">10<tspan baseline-shift="4" font-size="75%">−6</tspan></text><path d="M60,174.0 L420,174.0" class="mute"/><text x="54" y="178.0" text-anchor="end" class="sm">10<tspan baseline-shift="4" font-size="75%">−4</tspan></text><path d="M60,124.0 L420,124.0" class="mute"/><text x="54" y="128.0" text-anchor="end" class="sm">10<tspan baseline-shift="4" font-size="75%">−2</tspan></text><path d="M60,74.0 L420,74.0" class="mute"/><text x="54" y="78.0" text-anchor="end" class="sm">10<tspan baseline-shift="4" font-size="75%">0</tspan></text><path d="M60,24.0 L420,24.0" class="mute"/><text x="54" y="28.0" text-anchor="end" class="sm">10<tspan baseline-shift="4" font-size="75%">2</tspan></text><text x="60.0" y="240" text-anchor="middle" class="sm">0</text><text x="120.0" y="240" text-anchor="middle" class="sm">20</text><text x="180.0" y="240" text-anchor="middle" class="sm">40</text><text x="240.0" y="240" text-anchor="middle" class="sm">60</text><text x="300.0" y="240" text-anchor="middle" class="sm">80</text><text x="360.0" y="240" text-anchor="middle" class="sm">100</text><text x="420.0" y="240" text-anchor="middle" class="sm">120</text><text x="240.0" y="256" text-anchor="middle" class="sm">yineleme n</text><path d="M60.0,24.4 L63.0,32.7 L66.0,34.0 L69.0,35.2 L72.0,36.4 L75.0,37.5 L78.0,38.6 L81.0,39.8 L84.0,40.9 L87.0,42.1 L90.0,43.2 L93.0,44.4 L96.0,45.5 L99.0,46.7 L102.0,47.8 L105.0,48.9 L108.0,50.1 L111.0,51.2 L114.0,52.4 L117.0,53.5 L120.0,54.7 L123.0,55.8 L126.0,57.0 L129.0,58.1 L132.0,59.2 L135.0,60.4 L138.0,61.5 L141.0,62.7 L144.0,63.8 L147.0,65.0 L150.0,66.1 L153.0,67.2 L156.0,68.4 L159.0,69.5 L162.0,70.7 L165.0,71.8 L168.0,73.0 L171.0,74.1 L174.0,75.3 L177.0,76.4 L180.0,77.5 L183.0,78.7 L186.0,79.8 L189.0,81.0 L192.0,82.1 L195.0,83.3 L198.0,84.4 L201.0,85.6 L204.0,86.7 L207.0,87.8 L210.0,89.0 L213.0,90.1 L216.0,91.3 L219.0,92.4 L222.0,93.6 L225.0,94.7 L228.0,95.8 L231.0,97.0 L234.0,98.1 L237.0,99.3 L240.0,100.4 L243.0,101.6 L246.0,102.7 L249.0,103.9 L252.0,105.0 L255.0,106.1 L258.0,107.3 L261.0,108.4 L264.0,109.6 L267.0,110.7 L270.0,111.9 L273.0,113.0 L276.0,114.2 L279.0,115.3 L282.0,116.4 L285.0,117.6 L288.0,118.7 L291.0,119.9 L294.0,121.0 L297.0,122.2 L300.0,123.3 L303.0,124.4 L306.0,125.6 L309.0,126.7 L312.0,127.9 L315.0,129.0 L318.0,130.2 L321.0,131.3 L324.0,132.5 L327.0,133.6 L330.0,134.7 L333.0,135.9 L336.0,137.0 L339.0,138.2 L342.0,139.3 L345.0,140.5 L348.0,141.6 L351.0,142.7 L354.0,143.9 L357.0,145.0 L360.0,146.2 L363.0,147.3 L366.0,148.5 L369.0,149.6 L372.0,150.8 L375.0,151.9 L378.0,153.0 L381.0,154.2 L384.0,155.3 L387.0,156.5 L390.0,157.6 L393.0,158.8 L396.0,159.9 L399.0,161.1 L402.0,162.2 L405.0,163.3 L408.0,164.5 L411.0,165.6 L414.0,166.8 L417.0,167.9 L420.0,169.1" class="ca w2" stroke-opacity="0.55"/><path d="M60.0,42.4 L63.0,45.0 L66.0,46.1 L69.0,47.7 L72.0,48.9 L75.0,50.3 L78.0,51.5 L81.0,52.6 L84.0,53.8 L87.0,54.9 L90.0,56.1 L93.0,57.2 L96.0,58.4 L99.0,59.5 L102.0,60.7 L105.0,61.8 L108.0,62.9 L111.0,64.1 L114.0,65.2 L117.0,66.4 L120.0,67.5 L123.0,68.7 L126.0,69.8 L129.0,71.0 L132.0,72.1 L135.0,73.2 L138.0,74.4 L141.0,75.5 L144.0,76.7 L147.0,77.8 L150.0,79.0 L153.0,80.1 L156.0,81.2 L159.0,82.4 L162.0,83.5 L165.0,84.7 L168.0,85.8 L171.0,87.0 L174.0,88.1 L177.0,89.3 L180.0,90.4 L183.0,91.5 L186.0,92.7 L189.0,93.8 L192.0,95.0 L195.0,96.1 L198.0,97.3 L201.0,98.4 L204.0,99.6 L207.0,100.7 L210.0,101.8 L213.0,103.0 L216.0,104.1 L219.0,105.3 L222.0,106.4 L225.0,107.6 L228.0,108.7 L231.0,109.8 L234.0,111.0 L237.0,112.1 L240.0,113.3 L243.0,114.4 L246.0,115.6 L249.0,116.7 L252.0,117.9 L255.0,119.0 L258.0,120.1 L261.0,121.3 L264.0,122.4 L267.0,123.6 L270.0,124.7 L273.0,125.9 L276.0,127.0 L279.0,128.2 L282.0,129.3 L285.0,130.4 L288.0,131.6 L291.0,132.7 L294.0,133.9 L297.0,135.0 L300.0,136.2 L303.0,137.3 L306.0,138.4 L309.0,139.6 L312.0,140.7 L315.0,141.9 L318.0,143.0 L321.0,144.2 L324.0,145.3 L327.0,146.5 L330.0,147.6 L333.0,148.7 L336.0,149.9 L339.0,151.0 L342.0,152.2 L345.0,153.3 L348.0,154.5 L351.0,155.6 L354.0,156.7 L357.0,157.9 L360.0,159.0 L363.0,160.2 L366.0,161.3 L369.0,162.5 L372.0,163.6 L375.0,164.8 L378.0,165.9 L381.0,167.0 L384.0,168.2 L387.0,169.3 L390.0,170.5 L393.0,171.6 L396.0,172.8 L399.0,173.9 L402.0,175.1 L405.0,176.2 L408.0,177.3 L411.0,178.5 L414.0,179.6 L417.0,180.8 L420.0,181.9" class="ca w2" stroke-opacity="0.55"/><path d="M60.0,34.2 L63.0,40.7 L66.0,46.5 L69.0,47.7 L72.0,50.3 L75.0,51.5 L78.0,52.8 L81.0,53.9 L84.0,55.1 L87.0,56.2 L90.0,57.4 L93.0,58.5 L96.0,59.7 L99.0,60.8 L102.0,62.0 L105.0,63.1 L108.0,64.2 L111.0,65.4 L114.0,66.5 L117.0,67.7 L120.0,68.8 L123.0,70.0 L126.0,71.1 L129.0,72.2 L132.0,73.4 L135.0,74.5 L138.0,75.7 L141.0,76.8 L144.0,78.0 L147.0,79.1 L150.0,80.3 L153.0,81.4 L156.0,82.5 L159.0,83.7 L162.0,84.8 L165.0,86.0 L168.0,87.1 L171.0,88.3 L174.0,89.4 L177.0,90.5 L180.0,91.7 L183.0,92.8 L186.0,94.0 L189.0,95.1 L192.0,96.3 L195.0,97.4 L198.0,98.6 L201.0,99.7 L204.0,100.8 L207.0,102.0 L210.0,103.1 L213.0,104.3 L216.0,105.4 L219.0,106.6 L222.0,107.7 L225.0,108.9 L228.0,110.0 L231.0,111.1 L234.0,112.3 L237.0,113.4 L240.0,114.6 L243.0,115.7 L246.0,116.9 L249.0,118.0 L252.0,119.1 L255.0,120.3 L258.0,121.4 L261.0,122.6 L264.0,123.7 L267.0,124.9 L270.0,126.0 L273.0,127.2 L276.0,128.3 L279.0,129.4 L282.0,130.6 L285.0,131.7 L288.0,132.9 L291.0,134.0 L294.0,135.2 L297.0,136.3 L300.0,137.5 L303.0,138.6 L306.0,139.7 L309.0,140.9 L312.0,142.0 L315.0,143.2 L318.0,144.3 L321.0,145.5 L324.0,146.6 L327.0,147.7 L330.0,148.9 L333.0,150.0 L336.0,151.2 L339.0,152.3 L342.0,153.5 L345.0,154.6 L348.0,155.8 L351.0,156.9 L354.0,158.0 L357.0,159.2 L360.0,160.3 L363.0,161.5 L366.0,162.6 L369.0,163.8 L372.0,164.9 L375.0,166.0 L378.0,167.2 L381.0,168.3 L384.0,169.5 L387.0,170.6 L390.0,171.8 L393.0,172.9 L396.0,174.1 L399.0,175.2 L402.0,176.3 L405.0,177.5 L408.0,178.6 L411.0,179.8 L414.0,180.9 L417.0,182.1 L420.0,183.2" class="ca w2" stroke-opacity="0.55"/><path d="M60.0,40.6 L63.0,42.9 L66.0,44.1 L69.0,45.3 L72.0,46.5 L75.0,47.6 L78.0,48.8 L81.0,49.9 L84.0,51.1 L87.0,52.2 L90.0,53.3 L93.0,54.5 L96.0,55.6 L99.0,56.8 L102.0,57.9 L105.0,59.1 L108.0,60.2 L111.0,61.3 L114.0,62.5 L117.0,63.6 L120.0,64.8 L123.0,65.9 L126.0,67.1 L129.0,68.2 L132.0,69.4 L135.0,70.5 L138.0,71.6 L141.0,72.8 L144.0,73.9 L147.0,75.1 L150.0,76.2 L153.0,77.4 L156.0,78.5 L159.0,79.7 L162.0,80.8 L165.0,81.9 L168.0,83.1 L171.0,84.2 L174.0,85.4 L177.0,86.5 L180.0,87.7 L183.0,88.8 L186.0,89.9 L189.0,91.1 L192.0,92.2 L195.0,93.4 L198.0,94.5 L201.0,95.7 L204.0,96.8 L207.0,98.0 L210.0,99.1 L213.0,100.2 L216.0,101.4 L219.0,102.5 L222.0,103.7 L225.0,104.8 L228.0,106.0 L231.0,107.1 L234.0,108.2 L237.0,109.4 L240.0,110.5 L243.0,111.7 L246.0,112.8 L249.0,114.0 L252.0,115.1 L255.0,116.3 L258.0,117.4 L261.0,118.5 L264.0,119.7 L267.0,120.8 L270.0,122.0 L273.0,123.1 L276.0,124.3 L279.0,125.4 L282.0,126.6 L285.0,127.7 L288.0,128.8 L291.0,130.0 L294.0,131.1 L297.0,132.3 L300.0,133.4 L303.0,134.6 L306.0,135.7 L309.0,136.8 L312.0,138.0 L315.0,139.1 L318.0,140.3 L321.0,141.4 L324.0,142.6 L327.0,143.7 L330.0,144.9 L333.0,146.0 L336.0,147.1 L339.0,148.3 L342.0,149.4 L345.0,150.6 L348.0,151.7 L351.0,152.9 L354.0,154.0 L357.0,155.1 L360.0,156.3 L363.0,157.4 L366.0,158.6 L369.0,159.7 L372.0,160.9 L375.0,162.0 L378.0,163.2 L381.0,164.3 L384.0,165.4 L387.0,166.6 L390.0,167.7 L393.0,168.9 L396.0,170.0 L399.0,171.2 L402.0,172.3 L405.0,173.5 L408.0,174.6 L411.0,175.7 L414.0,176.9 L417.0,178.0 L420.0,179.2" class="cb w2" stroke-opacity="1"/><path d="M60.0,24.39151901920251 L420.0,161.66399070122785" stroke="currentColor" stroke-dasharray="5 4" fill="none"/><text x="440" y="38" class="sm">‖H*ⁿṼ − H*ⁿV‖∞, γ = 0.9</text><text x="440" y="56" class="sm">mavi: üç rastgele başlangıç çifti</text><text x="440" y="74" class="sm">turuncu: V₀ = 0 ile V* arası</text><text x="440" y="92" class="sm">kesikli: γⁿ eğimi (sınır)</text><text x="440" y="110" class="sm"></text><text x="440" y="128" class="sm">adım başına en büyük oran: 0.9000</text><text x="440" y="146" class="sm">(sınır γ = 0.9)</text><text x="440" y="164" class="sm"></text><text x="440" y="182" class="sm">V* = (-21.7, -18.5, -16.2, -16.7)</text><text x="440" y="200" class="sm">durağan politika (S1, S2, S3):</text><text x="440" y="218" class="sm">  yüksek, yüksek, indirim</text></svg><figcaption>Şekil 2 — Büzülme: aynı operatör iki farklı vektöre uygulandıkça aradaki ∞-norm uzaklığı her adımda en çok γ katına iner (perakende MDP'sinin sonsuz ufuklu biçimi, γ = 0.9; hesaplandı). Hangi vektörden başlanırsa başlansın aynı sabit noktaya, V*'a varılır.</figcaption></figure>

**Teorem 3.** $\gamma<1$ ise her $V$ için $\lim_nH^{*(n)}(V)=V^*$'dır. Kanıt Teorem 1'in aynısıdır, Lemma 1 yerine Lemma 2 kullanılır. Pratik algoritma bir durma kuralı ekler:

```text
V ← max_a R_a,  n ← 0
tekrarla:
    n ← n + 1
    V_n ← max_a [ R_a + γ T^a V_{n−1} ]
‖V_n − V_{n−1}‖∞ ≤ ε olana dek
döndür V_n
```

Teorem 2'nin aynı türetimi burada da geçerlidir: $\|V_n-V^*\|_\infty\le\gamma\epsilon/(1-\gamma)$. Kaç yineleme gerekir? Başlangıç hatası en çok $R_{\max}/(1-\gamma)$ olduğuna göre hatanın $\delta$'nın altına inmesi için $\gamma^n R_{\max}/(1-\gamma)\le\delta$, yani $n\ge\ln\bigl(\delta(1-\gamma)/R_{\max}\bigr)/\ln\gamma$ yeter. $\gamma=0.99$'da yineleme sayısı $\gamma=0.9$'dakinin yaklaşık on katıdır, çünkü $\ln0.9/\ln0.99\approx10.5$'tir.

### Çıkarılan politika ne kadar iyi? {#politika-siniri}

Değerlerin yaklaşık olması bir şeydir, onlardan çıkarılan politikanın kalitesi başka. $V_n$'ye göre açgözlü politika
$$\pi_n(s)=\arg\max_a\Bigl[R(s,a)+\gamma\sum_{s^+}P(s^+\mid s,a)V_n(s^+)\Bigr]$$
olsun. Asıl soru $\|V^{\pi_n}-V^*\|_\infty$'un, yani bu politikayı **gerçekten izlemenin** kaybının ne kadar olduğudur. Kaynak Teorem 4'ü $2\epsilon/(1-\gamma)$ sınırıyla veriyor, ama kanıt slaytı boş. Kanıt aşağıdadır ve sıkı sınır bir $\gamma$ çarpanı daha küçüktür.

:::türetim Teorem 4: $\|V^{\pi_n}-V^*\|_\infty\le2\gamma\epsilon/(1-\gamma)$
$H_\pi(V)=R_\pi+\gamma T^\pi V$ ve $\|V_n-V_{n-1}\|_\infty\le\epsilon$ olsun. $\pi_n$, $V_n$'ye göre açgözlü olduğu için $H_{\pi_n}V_n=H^*V_n$ olur. İki parça ayrı sınırlanır.

1. **$V^{\pi_n}$ ile $V_n$ arası.** $V^{\pi_n}=H_{\pi_n}V^{\pi_n}$ olduğundan
$$\|V^{\pi_n}-V_n\|\le\|H_{\pi_n}V^{\pi_n}-H_{\pi_n}V_n\|+\|H^*V_n-V_n\|\le\gamma\|V^{\pi_n}-V_n\|+\gamma\epsilon$$
olur. Son terim $\|H^*V_n-H^*V_{n-1}\|\le\gamma\|V_n-V_{n-1}\|$'den gelir. Buradan $\|V^{\pi_n}-V_n\|\le\gamma\epsilon/(1-\gamma)$ çıkar.
2. **$V^*$ ile $V_n$ arası.** Teorem 2'nin değer yinelemesi sürümünden $\|V^*-V_n\|\le\gamma\epsilon/(1-\gamma)$'dır.

Üçgen eşitsizliğiyle $\|V^{\pi_n}-V^*\|_\infty\le2\gamma\epsilon/(1-\gamma)$ elde edilir. Notlardaki $2\epsilon/(1-\gamma)$ bunun gevşek biçimidir. Puterman'ın (Teorem 6.3.1) pratik kuralı buradan çıkar: $\epsilon$-en iyi bir politika için $\|V_{n}-V_{n-1}\|_\infty<\epsilon(1-\gamma)/(2\gamma)$ olana dek yinelemek yeter.
:::

Pratikte sınır kötümserdir. Açgözlü politika çoğu zaman değerler yakınsamadan çok önce en iyi politikaya oturur (aşağıdaki deney). Bu oturma tekdüze olmak zorunda değildir. Deneyin varsayılan ayarında politika 4. yinelemede ilk kez en iyi olur, sonra yeniden sapar ve ancak 17. yinelemeden itibaren kalıcı olarak en iyi kalır. Değerlerin durma eşiğine ulaşması ise 65 yineleme sürer. Sonlu bir MDP'de politika sayısı sonludur ve yeterince küçük bir değer hatasında açgözlü politika tam olarak en iyidir. M11.3'teki politika yinelemesi bu gözlemden yararlanır: değerleri değil politikayı yineler ve sonlu adımda biter.

::lab l3

**Karmaşıklık.** Her yinelemede her eylem için bir $|\mathcal S|\times|\mathcal S|$ matris–vektör çarpımı ve öğe öğe bir maksimum yapılır: toplam $O(N|\mathcal A||\mathcal S|^2)$, $N$ yineleme sayısıdır. Geçişler seyrekse (her durumdan birkaç ardıl) bir yineleme $O(|\mathcal A|\cdot\text{sıfır olmayan geçiş})$'e iner. Büyük $\gamma$ hem $N$'yi büyütür hem sınırları gevşetir.

Aşağıdaki ızgara dünyası deneyi aynı kavramları bir yol bulma probleminde gösteriyor: değerlerin farkı log ölçekte eğimi $\ln\gamma$ olan bir doğru çizer. Kayma olasılığı ve adım ödülü politikanın ne kadar risk aldığını belirler.

::lab l1

### Özet {#ozet}

- MDP $(\mathcal S,\mathcal A,P,R,\gamma,h)$ altılısıdır; politika durumdan eyleme bir eşlemedir. Planlamada model bilinir, ama geçişler rastgele olabilir.
- Durum tanımı Markov özelliğini belirler. Robot süpürgede konum duruma eklenmeden "öbür odaya geç" eylemi modellenemez.
- İndirimli toplam $|R|\le R_{\max}$ ve $\gamma<1$ ise $R_{\max}/(1-\gamma)$ ile sınırlıdır; $\gamma=1$ yalnız sonlu ufukta ya da uygun politikalarda kullanılır.
- Beklenen fayda $t$ adımlık durum dağılımıyla tanımlanır. Değer fonksiyonu Bellman denklemiyle sondan başa hesaplanır: dinamik programlama $|\mathcal A|^h$ yolu $h|\mathcal S|^2|\mathcal A|$ işleme indirir.
- Perakende stok örneğinde en iyi eylem kalan güne göre değişir: sonlu ufukta politika durağan değildir, sonsuz ufukta durağandır.
- Matris biçimi: $V_t=\max_a[R_a+\gamma T^aV_{t-1}]$; $T^a$ satır-stokastiktir. Politika değerlendirme $V^\pi=(I-\gamma T^\pi)^{-1}R_\pi$'dir ve $\gamma<1$'de ters her zaman vardır.
- $H$ ve $H^*$, $\gamma<1$'de ∞-normda $\gamma$-büzülmedir; yineleme her başlangıçtan tek sabit noktaya yakınsar.
- Durma kuralı $\|V_n-V_{n-1}\|_\infty\le\epsilon$ ile değer hatası $\le\gamma\epsilon/(1-\gamma)$, açgözlü politikanın kaybı $\le2\gamma\epsilon/(1-\gamma)$'dır. Notlardaki sınırlar bir $\gamma$ çarpanı gevşektir. Politika çoğu zaman değerlerden çok önce en iyi olur.
- Karmaşıklık yineleme başına $O(|\mathcal A||\mathcal S|^2)$'dir.
