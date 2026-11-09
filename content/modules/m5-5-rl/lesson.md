### Makine öğrenmesinin dört biçimi {#turler}

Makine öğrenmesi, bilgisayarların açıkça programlanmadan veriden öğrenmesini ve başarımını artırmasını sağlayan yapay zekâ alanıdır: algoritmalar veriyi analiz eder, örüntüler bulur, tahmin ya da karar üretir. Modeller veriyle dört biçimde eğitilir:
- **Denetimli öğrenme:** tüm veri etiketlidir. Model bir tahmin, karar ya da sınıflandırma üretir; ürettiği her sonuç için doğru cevabı veririz ve model çıktısını doğru cevapla karşılaştırarak iyileşir (Faz 1–4).
- **Yarı denetimli öğrenme:** veri kısmen etiketlidir. Etiketli kısım denetimli öğrenmedeki gibi kullanılır, geri kalanı denetimsiz biçimde (ör. M5.4'teki GCN'in 140 etiketle tüm grafı kullanması).
- **Denetimsiz öğrenme:** veri etiketsizdir. Model veriyi kümelemek için eğitilir; probleme özgü bir benzerlik ölçütüyle birbirine benzeyen örnekleri öğrenmeye çalışır (M2.7). Bir başka yaklaşım üretken stratejidir: veri bilerek kısmen bozulur ya da silinir ve model eksik parçaları geri kurmak için eğitilir (M4.16'daki BERT ve BART, M5.1'deki otomatik kodlayıcı).
- **Pekiştirmeli öğrenme** (reinforcement learning, RL): bu modülün ve Faz 11'in konusu.

Modern makine öğrenmesi modellerinin son başarıları büyük ölçüde veriden gelir; model büyüdükçe daha çok veri gerekir. Denetimli öğrenme en dolaysız yaklaşımdır. Ama eğitim verisi arttıkça etiketleme son derece ciddi bir sorun hâline gelir: yavaş, pahalı ve hataya açıktır. ChatGPT ve Stable Diffusion gibi modern üretken modeller bu sorunu denetimsiz (kendinden denetimli) öğrenmeye geçerek aştı. Slaytın son notu ise şudur: en ileri modeller o kadar büyüdü ki onları eğitmeye yetecek kadar veri (etiketli ya da etiketsiz) bulmak zorlaşıyor. Bir ajanın kendi deneyimini üretmesi (RL) bu açıdan cazip bir alternatiftir.

:::uyarı Kaynak notu: yazım
Slayt 6'da "Reinforced learning" yazıyor; doğrusu **reinforcement learning**'dir (pekiştirmeli öğrenme). Slayt 14'teki "DeepmMind" da DeepMind olmalıdır.
:::

### Pekiştirmeli öğrenme kurgusu {#kurgu}

Pekiştirmeli öğrenme belirli bir kurgu varsayar. Probleminiz, belirli kuralları olan ve durumu (belki kısmen) gözlenebilen etkileşimli bir ortam olarak tarif edilebiliyorsa, onu ödülleri en büyüklemeyi amaçlayan bir **oyun** olarak kurabilirsiniz:
- Model **oyuncu** (ajan) olur. Etkileşimli ortamın şimdiki **durumuna** bakar, işler ve bir çıktı üretir.
- Bu çıktı ortama bir **eylem** olarak verilir.
- Ortam eylemi işler ve modelin çıktısının kalitesini anlatan bir **ödül** (ya da ceza) döndürür.
- Oyun ya bir son duruma (kazanç ya da kayıp) varılana kadar ya da probleme göre belirlenmiş sayıda tur oynanır.

<figure class="fig wide"><svg viewBox="0 0 640 230" role="img" aria-label="Ajan ortamı gözler ve bir eylem seçer; ortam yeni durumu ve bir ödülü döndürür; amaç beklenen indirimli getiriyi en büyüklemektir"><defs><marker id="r1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker><marker id="r2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker></defs><rect x="60" y="70" width="150" height="60" rx="8" class="soft-a"/><text x="135" y="98" text-anchor="middle" class="ta">ajan</text><text x="135" y="116" text-anchor="middle" class="sm">politika π(a | s)</text><rect x="400" y="70" width="170" height="60" rx="8" class="soft-p"/><text x="485" y="98" text-anchor="middle" class="tp">ortam</text><text x="485" y="116" text-anchor="middle" class="sm">P(s′ | s, a), R(s, a)</text><path d="M210,85 L400,85" class="cb w2" marker-end="url(#r1)"/><text x="305" y="78" text-anchor="middle" class="tb">eylem a<tspan baseline-shift="-3" font-size="75%">t</tspan></text><path d="M400,115 L210,115" class="ca w2" marker-end="url(#r2)"/><text x="305" y="134" text-anchor="middle" class="ta">durum s<tspan baseline-shift="-3" font-size="75%">t+1</tspan>, ödül r<tspan baseline-shift="-3" font-size="75%">t+1</tspan></text><text x="20" y="176" class="sm">Atari: durum = ekran görüntüsü, eylem = joystick, ödül = skor değişimi. Go: durum = tahta, eylem = taşın yeri, ödül = ±1 (oyun sonunda).</text><text x="20" y="194" class="sm">Ticaret: durum = fiyat geçmişi, eylem = al/sat/tut, ödül = kâr. Öneri: durum = izleme geçmişi, eylem = öneri, ödül = memnuniyet.</text><text x="20" y="218" class="sm">Amaç: indirimli getiri G<tspan baseline-shift="-3" font-size="75%">t</tspan> = r<tspan baseline-shift="-3" font-size="75%">t+1</tspan> + γ r<tspan baseline-shift="-3" font-size="75%">t+2</tspan> + γ² r<tspan baseline-shift="-3" font-size="75%">t+3</tspan> + … beklentisini en büyüklemek.</text></svg><figcaption>Şekil 1 — Pekiştirmeli öğrenmenin etkileşim döngüsü. Ajan yalnız ödülden geri bildirim alır: doğru eylem söylenmez (denetimli öğrenmeden farkı), ödül gecikmeli ve gürültülü olabilir, ajanın kararları gelecekte göreceği verileri değiştirir.</figcaption></figure>

Pekiştirmeli öğrenme, davranışçı psikolojiden esinlenmiş bir makine öğrenmesi alanıdır: yazılım ajanlarının bir ortamda birikimli ödülü en büyüklemek için nasıl eylem seçmesi gerektiğiyle ilgilenir. Başka alanlarda başka adlarla bilinir: kontrol kuramında **optimal kontrol**, optimizasyon kuramında **yaklaşık dinamik programlama**, yöneylem araştırmasında **nöro-dinamik programlama**. Çoğu durumda mesele **stokastik kontroldür**. Bu adlar RL'nin ne kadar yaygın olduğunu gösterir.

RL 1950'lerden beri vardır ve yıllar içinde ilginç uygulamalar üretmiştir: oyunlarda (ör. tavla oynayan TD-Gammon; Tesauro 1992) ve makine kontrolünde. Ama nadiren manşet olmuştur. 2013'te bir devrim yaşandı: İngiliz girişimi DeepMind, hemen her Atari oyununu sıfırdan oynamayı öğrenen ve sonunda çoğunda insanları geçen bir sistem gösterdi. Sistem girdi olarak yalnız ham pikselleri kullanıyordu ve oyunların kuralları hakkında önceden hiçbir bilgisi yoktu (Mnih ve ark. 2013; ayrıntılar M11.6).

**Hayvan psikolojisiyle bağlantı.** Hayvanlar pekiştirmelerle eğitilir: istenen davranıştan sonra yiyecek ya da haz (olumlu pekiştirme). RL aynı stratejiyi makine öğrenmesi modellerine uygular. RL, denetimli ve denetimsiz öğrenmeden temelden farklıdır: kimse ajana doğru eylemi söylemez.

:::uyarı Kaynak notu: "negatif pekiştirme"
Slayt 12 acıyı ve açlığı **negatif pekiştirme** örneği olarak sayıyor. Davranışçı psikolojinin terimlerinde negatif pekiştirme, hoş olmayan bir uyaranın **kaldırılmasıyla** davranışın artmasıdır (ör. emniyet kemerini takınca uyarı sesinin susması). Acı vermek, davranışı azaltan bir **cezadır** (olumlu ceza). RL açısından ayrım şudur: ödül (pozitif skaler) davranışı artırır, ceza (negatif skaler) azaltır. "Negatif ödül" demek daha az karışıklık yaratır.
:::

### Örnekler {#ornekler}

| örnek | ajan | ortam | durum | eylem | ödül |
|---|---|---|---|---|---|
| DeepMind Atari | oyuncu | oyun kuralları ve konsol | ekran görüntüsü | sonraki joystick hareketi | skor değişimi |
| AlphaGo | oyuncu | kurallar ve rakip | tahta düzeni | taşın konacağı kesişim | oyun sonunda +1 / −1 |
| otomatik ticaret | ticaret yazılımı | piyasa ve diğer yatırımcılar | fiyat geçmişi | al / sat / tut | kâr |
| öneri sistemi | öneri yazılımı | izleyici | izleme geçmişi | sonraki öneri(ler) | izleyici memnuniyeti |

:::uyarı Kaynak düzeltmesi: Atari ödülü ve Go'nun durum sayısı
- Slayt 14 Atari için ödülü "+1 kazanma / −1 kaybetme" olarak veriyor. DQN'de ödül oyunun **skor değişimidir** (eğitimde işaretine göre +1, 0, −1'e kırpılır). Kazanma/kaybetme ödülü Go için geçerlidir.
- Slayt 16, 19 × 19 Go tahtasının olası durum sayısını kabaca $3^{361}\approx1.72\times10^{172}$ olarak veriyor. $361\log_{10}3=172.24$ olduğundan doğrusu $3^{361}\approx1.74\times10^{172}$'dir. Yasal pozisyonların gerçek sayısı (Tromp 2016) ≈ $2.08\times10^{170}$'tir; slayttaki "daha iyi yaklaşım 2.1 × 10¹⁷⁰" bununla uyumludur. Evrende tahminen $10^{80}$ atom vardır.
:::

AlphaGo 2016'da en iyi Go oyuncularından Lee Sedol'u 4–1 yendi. Slaytın vurgusu: 2. oyundaki ünlü 37. hamle, denetimli bir kurguda öğrenilmesi muhtemelen imkânsız bir hamleydi; uzmanların hamlelerini taklit eden bir model böyle bir hamleyi neredeyse hiç önermezdi (ayrıntılar M11.11).

**RL'nin zorlukları ve fırsatları** (slayt 20):
- Doğrudan denetim yoktur (denetimli öğrenmedeki gibi doğru cevap verilmez).
- Geri bildirim kısmi ve **gecikmelidir**: ödüller tipik olarak anında gelmez; hangi eylemin ödüle yol açtığını bulmak gerekir (kredi atama).
- Ortamlar stokastiktir; ama gerçek hayattaki problemlerin çoğu da stokastiktir.
- Model eksik olabilir: tüm durumları, eylemleri ya da ödülleri bilmeyebilir ya da göremeyebiliriz.
- Kararlar birbirine bağlıdır: bugünkü eylem yarın görülecek durumu değiştirir (Markov varsayımı).

Slayta göre RL, ömür boyu öğrenen makinelerin en gerçekçi yoludur.

### MDP ve Bellman denklemleri {#mdp}

Etkileşimi matematiksel olarak kurmak için **Markov karar süreci** (MDP) kullanılır. Bu modülde temelleri kuruyoruz; rastgele süreçler, Markov zincirleri, değer ve politika iterasyonunun yakınsama kanıtları Faz 11'de (M11.1–M11.3).

:::tanım MDP, getiri, değer
- **MDP:** durumlar $\mathcal S$, eylemler $\mathcal A$, geçiş olasılıkları $P(s'\mid s,a)$, ödül $R(s,a)$, indirim $\gamma\in[0,1)$. **Markov özelliği:** bir sonraki durum yalnız şimdiki durum ve eyleme bağlıdır, geçmişe değil.
- **Politika:** $\pi(a\mid s)$, durumdan eylem olasılıklarına bir eşleme (deterministik ya da stokastik).
- **Getiri:** $G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\dots=\sum_{k\ge0}\gamma^kr_{t+k+1}$. $\gamma<1$ sonsuz toplamı sonlu tutar ve yakın ödülleri uzak ödüllerden değerli sayar.
- **Durum değeri:** $V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s]$. **Eylem değeri:** $Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a]$.
:::

:::türetim Bellman beklenti denklemi ve kapalı çözümü
Getiri özyinelidir: $G_t=r_{t+1}+\gamma G_{t+1}$. Koşullu beklenti alalım ve Markov özelliğini kullanalım:
$$V^\pi(s)=\sum_a\pi(a\mid s)\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^\pi(s')\Big].$$
Sonlu durumlu bir MDP'de bu bir doğrusal denklem sistemidir. $\mathbf r_\pi(s)=\sum_a\pi(a\mid s)R(s,a)$ ve $\mathbf P_\pi(s,s')=\sum_a\pi(a\mid s)P(s'\mid s,a)$ ile:
$$\mathbf V^\pi=\mathbf r_\pi+\gamma\mathbf P_\pi\mathbf V^\pi\quad\Rightarrow\quad\mathbf V^\pi=(\mathbf I-\gamma\mathbf P_\pi)^{-1}\mathbf r_\pi .$$
Ters, $\gamma<1$ ve $\mathbf P_\pi$ stokastik olduğu için vardır: $\gamma\mathbf P_\pi$'nin spektral yarıçapı en fazla $\gamma<1$'dir ve Neumann serisi $\sum_k(\gamma\mathbf P_\pi)^k$ yakınsar. Seri, "$k$ adım sonraki beklenen ödülün $\gamma^k$ ağırlıklı toplamı"nın matris hâlidir.
:::

:::türetim Bellman optimallik denklemi ve değer iterasyonu
En iyi değer $V^*(s)=\max_\pi V^\pi(s)$ şu denklemi sağlar:
$$V^*(s)=\max_a\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\Big].$$
**Değer iterasyonu** bu denklemi sabit nokta iterasyonu olarak uygular: $V_{k+1}=\mathcal T V_k$. **Neden yakınsar?** $\mathcal T$, sup normunda bir **$\gamma$-büzülmedir**. Herhangi iki $V$, $U$ için $|\max_af(a)-\max_ag(a)|\le\max_a|f(a)-g(a)|$ olduğundan:
$$|\mathcal TV(s)-\mathcal TU(s)|\le\max_a\gamma\sum_{s'}P(s'\mid s,a)|V(s')-U(s')|\le\gamma\|V-U\|_\infty .$$
Banach sabit nokta teoremine göre tek bir sabit nokta ($V^*$) vardır ve $\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty$. En iyi politika açgözlüdür: $\pi^*(s)=\arg\max_a Q^*(s,a)$.
:::

:::örnek El hesabı (modülün el hesabı alanları)
İki durum, deterministik. $s_0$'da "kal" ödül 0 verir ve $s_0$'da kalır; "git" ödül 1 verir ve $s_1$'e gider. $s_1$'de "kal" ödül 2 verir; "git" ödül 0 verir ve $s_0$'a döner. $\gamma=0.5$, $V_0=0$.
- $V_1(s_0)=\max(0,1)=\mathbf 1$, $V_1(s_1)=\max(2,0)=2$.
- $V_2(s_0)=\max(0+0.5\cdot1,\ 1+0.5\cdot2)=\mathbf 2$, $V_2(s_1)=\max(2+0.5\cdot2,\ 0+0.5\cdot1)=\mathbf 3$.
- Limit: $s_1$'de hep kal, $V^*(s_1)=2/(1-0.5)=\mathbf 4$; $s_0$'da git, $V^*(s_0)=1+0.5\cdot4=\mathbf 3$; $Q^*(s_0,\text{kal})=0+0.5\cdot3=\mathbf{1.5}$.

"Git"in anlık ödülü küçüktür ama daha değerli bir duruma götürür; değer fonksiyonu gelecekteki ödülleri bugüne taşır.
:::

Aşağıdaki deneyde iki durumlu bir MDP'yi değer iterasyonuyla çöz. Hatanın $\gamma^k$ sınırının altında kaldığını gör; kayma olasılığıyla ödülleri değiştirip en iyi politikanın nasıl değiştiğini izle.

::lab l1

### Modelsiz öğrenme: değer tabanlı ve politika tabanlı {#modelsiz}

Değer iterasyonu $P$ ve $R$'yi bilmeyi gerektirir. Gerçek problemlerde model çoğu zaman bilinmez; ajan ortamla etkileşerek öğrenmelidir. İki ana aile vardır:
- **Değer tabanlı, modelsiz** (Q-öğrenme, DQN; M11.5, M11.6): $Q^*$'ı örneklerden kestirir; politika açgözlü olarak türetilir, ayrıca gösterilmez.
- **Politika tabanlı, modelsiz** (politika gradyanı; lect_8): politikayı doğrudan parametreler ve en iyiler; açık bir değer fonksiyonu tutmaz.

**Stokastik politika.** $\pi_\theta(a\mid s)=P(a\mid s;\theta)$, $\theta$ ile parametrelenmiş bir fonksiyondur. Duruma göre tek bir eylem seçmez; mevcut eylemler üzerinde bir olasılık dağılımı tanımlar.
- **Sonlu sayıda ayrık eylem:** softmax, $\pi_\theta(a\mid s)=\frac{\exp h(s,a;\theta)}{\sum_{a'}\exp h(s,a';\theta)}$. $h$, $\theta$'da doğrusal olabilir ($h=\sum_i\theta_if_i(s,a)$, $f_i$ bir temel fonksiyon ailesi) ya da bir sinir ağı.
- **Sürekli eylemler:** Gauss, $\pi_\theta(a\mid s)=\mathcal N\big(a\mid\mu(s;\theta),\sigma(s;\theta)\big)$.

**Denetimli öğrenmeyle karşılaştırma** (lect_8, slayt 5–6). Bir uzmanın optimal durum–eylem çiftleri $\{(s_i,a_i^*)\}$ elimizde olsaydı, verinin log-olabilirliğini en büyüklerdik: $\theta^*=\arg\max_\theta\sum_i\log\pi_\theta(a_i^*\mid s_i)$. Bu, stokastik gradyan yükselişiyle $\theta\leftarrow\theta+\alpha\nabla_\theta\log\pi_\theta(a_n^*\mid s_n)$ olur (davranış klonlama). RL'de ise doğru eylemi bilmeyiz; elimizde durum–eylem–ödül üçlüleri vardır ve beklenen indirimli ödül toplamını en büyükleriz. Ortaya çıkan REINFORCE güncellemesi $\theta\leftarrow\theta+\alpha\gamma^nG_n\nabla_\theta\log\pi_\theta(a_n\mid s_n)$, denetimli güncellemenin **getiriyle ağırlıklı** hâlidir. Yorumu: daha büyük indirimli ödül toplamı getiren eylemlerin etkisi daha büyük olmalıdır.

### Politika gradyanı teoremi {#pg}

Slayt 7–8, **stokastik politika gradyanı teoremini** durağan durum dağılımıyla verir ($\nabla V\propto\sum_s\mu_\theta(s)\sum_aQ_\theta(s,a)\nabla\pi_\theta(a\mid s)$) ve biçimsel kanıtının oldukça zor olduğunu söyler. Bölümlü (episodik) durumda ise kısa ve öğretici bir kanıt vardır.

:::türetim Olabilirlik oranı (log-türev) hilesi
Bir bölümün yörüngesi $\tau=(s_0,a_0,r_1,s_1,\dots)$. Olasılığı
$$p_\theta(\tau)=p(s_0)\prod_t\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t).$$
Amaç $J(\theta)=\mathbb E_{\tau\sim p_\theta}[G(\tau)]=\int p_\theta(\tau)G(\tau)\,d\tau$. $\nabla p=p\,\nabla\log p$ özdeşliğiyle:
$$\nabla_\theta J=\int p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)\,G(\tau)\,d\tau=\mathbb E_\tau\big[\nabla_\theta\log p_\theta(\tau)\,G(\tau)\big].$$
$\log p_\theta(\tau)$'nun $\theta$'ya bağlı tek kısmı politika terimleridir; başlangıç dağılımı ve geçiş olasılıkları türevde **kaybolur**:
$$\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).$$
Bu yüzden yöntem **modelsizdir**: $P$'yi bilmeden, yalnız örnek yörüngelerle gradyan kestirilir. Bu, M5.1'deki skor fonksiyonu kestiricisinin aynısıdır.
:::

:::türetim Nedensellik ve taban çizgisi
**Nedensellik.** $t$ anındaki eylem, kendinden önceki ödülleri etkileyemez. $k<t$ için $\mathbb E[\nabla\log\pi(a_t\mid s_t)\,r_k]=0$'dır, çünkü $r_k$ verildiğinde $\mathbb E_{a_t\sim\pi}[\nabla\log\pi(a_t\mid s_t)]=\sum_a\nabla\pi(a\mid s_t)=\nabla1=0$. Bu terimleri atınca:
$$\nabla_\theta J=\mathbb E\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\Big],\qquad G_t=\sum_{k\ge t}\gamma^{k-t}r_{k+1}.$$
**Taban çizgisi.** $a_t$'ye bağlı olmayan herhangi bir $b(s_t)$ için aynı nedenle $\mathbb E[\nabla\log\pi(a_t\mid s_t)\,b(s_t)]=0$. $G_t$ yerine $G_t-b(s_t)$ kullanmak gradyanı **yansız** bırakır ama **varyansı** düşürebilir. İyi bir seçim $b(s)\approx V^\pi(s)$'dir; $G_t-V(s_t)$ farkı **avantajdır**: eylemin ortalamadan ne kadar iyi olduğu. Bu, actor–critic yöntemlerinin (M11.9) çıkış noktasıdır.

**$\gamma^t$ çarpanı.** Sutton ve Barto'nun bölümlü REINFORCE'u (lect_8'deki güncelleme) başlangıç durumunun değerini en büyüklediği için $\gamma^nG_n$ kullanır. Pratikteki çoğu uygulama bu $\gamma^n$'yi atar; o zaman gradyan, durum dağılımını yanlı ağırlıklar ama genellikle daha iyi çalışır.
:::

:::örnek El hesabı: iki kollu haydut
$\pi=\mathrm{softmax}(\theta)$, $\theta=(0,0)$, yani $\pi=(0.5,0.5)$. Örneklenen eylem 0, ödül $R=1$. $\nabla_\theta\log\pi(a)=\mathbf e_a-\boldsymbol\pi$, dolayısıyla $\partial\log\pi(0)/\partial\theta_0=1-0.5=0.5$ ve gradyan bileşeni $R\cdot0.5=\mathbf{0.5}$. Taban çizgisi $b=0.6$ ile $(1-0.6)\cdot0.5=\mathbf{0.2}$. Taban çizgisi, "beklenenden iyi mi?" sorusunu sorar: ödül beklentiden az olsaydı ($R<b$) güncelleme eylemin olasılığını **düşürürdü**.
:::

:::tanım REINFORCE (Williams 1992; Sutton ve Barto, Bölüm 13)
$\pi_\theta$'yı herhangi bir şekilde başlat. Her bölüm için:
1. $\pi_\theta$ ile bir bölüm üret: $s_0,a_0,r_1,\dots,s_{T-1},a_{T-1},r_T$.
2. Her adım $n=0,\dots,T-1$ için $G_n\leftarrow\sum_{k=n+1}^T\gamma^{k-n-1}r_k$ ve $\theta\leftarrow\theta+\alpha\gamma^nG_n\nabla_\theta\log\pi_\theta(a_n\mid s_n)$.

PyTorch'ta bir bölümün güncellemesi tek bir kaybın gradyanıdır: `loss = -(log_probs * (G - b)).sum()`. Getiriler aşama e'de 64 paralel ortam için tek bir matris çarpımıyla hesaplanıyor.
:::

**Neden stokastik politika?** Değer tabanlı yöntemlerin açgözlü politikası deterministiktir. Ajanın durumları ayırt edemediği (kısmen gözlenebilir) ortamlarda en iyi politika stokastik olabilir. Sutton ve Barto'nun kısa koridor örneği bunu çarpıcı biçimde gösterir.

<figure class="fig wide"><svg viewBox="0 0 640 180" role="img" aria-label="Eylemleri yer değiştirmiş kısa koridor: üç durum ve hedef, ikinci durumda sağ ve sol ters çalışır"><rect x="40" y="50" width="110" height="44" rx="6" class="soft-a"/><text x="95" y="77" text-anchor="middle" class="sm">1 (başlangıç)</text><rect x="170" y="50" width="110" height="44" rx="6" class="soft-b"/><text x="225" y="77" text-anchor="middle" class="sm">2 (ters!)</text><rect x="300" y="50" width="110" height="44" rx="6" class="soft-a"/><text x="355" y="77" text-anchor="middle" class="sm">3</text><rect x="430" y="50" width="110" height="44" rx="6" class="soft-p"/><text x="485" y="77" text-anchor="middle" class="sm">hedef</text><text x="40" y="128" class="sm">1: sağ → 2, sol → 1 (duvar).  2: sağ → 1, sol → 3 (eylemler ters).  3: sağ → hedef, sol → 2.  Her adım −1.</text><text x="40" y="150" class="sm">Tüm durumlarda özellik aynı: politika tek bir sayıdır, p = P(sağ). Kesin değer: J(0.5) = −12, J(0.59) = −11.66 (en iyi), J(0.3) = −16.2;</text><text x="40" y="168" class="sm">p = 0 ya da 1 olan her deterministik politika hedefe hiç varmaz (J = −∞). En iyi politika stokastiktir.</text></svg><figcaption>Şekil 2 — Sutton ve Barto'nun (2018) Örnek 13.1'i. Ajan durumları ayırt edemez (aynı özellik), bu yüzden değer tabanlı yöntemlerin açgözlü (deterministik) politikaları başarısız olur. Politika gradyanı stokastik politikayı doğrudan öğrenir ve p* ≈ 0.59'a yakınsar (l2).</figcaption></figure>

Aşağıdaki deneyde REINFORCE'u bu koridorda çalıştır. Taban çizgisinin gradyan gürültüsünü nasıl azalttığını ve politikanın kesin çözülen optimuma (p* ≈ 0.59) nasıl yakınsadığını gör.

::lab l2

**AlphaGo'ya bakış.** lect_8'in son kısmı politika gradyanının en ünlü uygulamasını anlatır. AlphaGo dört adımda kuruldu:
1. KGS Go sunucusundaki 30 milyon tahta düzeninden uzman hamlelerini taklit eden denetimli bir politika ağı;
2. bu ağın kendisine karşı oynayarak politika gradyanıyla iyileştirilmesi;
3. pozisyonun kazanma olasılığını tahmin eden bir değer ağı;
4. politika ve değer ağlarıyla yönlendirilen Monte Carlo ağaç araması.

AlphaGo, Mart 2016'da Lee Sedol'u, Mayıs 2017'de dünya şampiyonu Ke Jie'yi yendi. DeepMind 2010'da Londra'da Demis Hassabis, Mustafa Suleyman ve Shane Legg tarafından kuruldu; Google 2014'te satın aldı. Ayrıntılar ve AlphaGo Zero M11.11'de.

### PyTorch'ta {#pytorch}

| kavram | PyTorch | not |
|---|---|---|
| politika (ayrık) | `logits = net(s); dist = torch.distributions.Categorical(logits=logits)` | |
| eylem ve log-olasılık | `a = dist.sample(); logp = dist.log_prob(a)` | `sample` gradyan taşımaz; gradyan `log_prob`'dan |
| sürekli eylem | `dist = Normal(mu, std)`; `a = dist.sample()` | |
| getiriler | `G = discount_matrix @ r` ya da sondan başa döngü | aşama e |
| REINFORCE kaybı | `loss = -(logp * (G - b).detach()).sum()` | taban çizgisi gradyan almamalı |
| entropi bonusu | `loss -= beta * dist.entropy().mean()` | erken determinizmi önler |
| Bellman (matris) | `V = torch.linalg.solve(I - gamma * P_pi, r_pi)` | aşama c |
| değer iterasyonu | `V = (R + gamma * torch.einsum("sat,t->sa", P, V)).max(1).values` | aşama c |

### Özet {#ozet}

- Denetimli, yarı denetimli, denetimsiz ve pekiştirmeli öğrenme. RL'de doğru cevap verilmez; ajan ödülü en büyükler. Geri bildirim gecikmeli, ortam stokastik, kararlar birbirine bağlıdır.
- Slayt notları: "Reinforced learning" → reinforcement; Atari ödülü skor değişimidir; $3^{361}\approx1.74\times10^{172}$; "negatif pekiştirme" psikolojide uyaranın kaldırılmasıdır, acı vermek cezadır.
- MDP $(\mathcal S,\mathcal A,P,R,\gamma)$; getiri $G_t=\sum\gamma^kr_{t+k+1}$; $\mathbf V^\pi=(\mathbf I-\gamma\mathbf P_\pi)^{-1}\mathbf r_\pi$.
- Bellman optimallik operatörü $\gamma$-büzülmedir; değer iterasyonu $\|V_k-V^*\|\le\gamma^k\|V_0-V^*\|$ ile yakınsar. El örneği: $V^*=(3,4)$, $Q^*(s_0,\text{kal})=1.5$.
- Politika gradyanı: $\nabla J=\mathbb E[\sum_t\nabla\log\pi(a_t\mid s_t)(G_t-b(s_t))]$; geçiş olasılıkları türevde kaybolur (modelsiz); taban çizgisi yansızdır ve varyansı düşürür. El örneği: 0.5 ve 0.2.
- Durumları ayıramayan ajanda en iyi politika stokastik olabilir (kısa koridor, p* ≈ 0.59, J* ≈ −11.66). Ayrıntılı RL Faz 11'de.
