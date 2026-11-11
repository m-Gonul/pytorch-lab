### Modeli bilmeden öğrenmek {#modelsiz}

M11.2 ve M11.3'te geçiş modeli $P(s^+\mid s,a)$ ve ödül fonksiyonu $R(s,a)$ biliniyordu. Değer ve politika yinelemesi bu modeli kullanarak en iyi politikayı **hesaplıyordu** (planlama). Pekiştirmeli öğrenmede bileşenler aynıdır: durumlar, eylemler, ödüller, politika, indirim ve ufuk. Fark, geçiş ve ödül modelinin **bilinmemesidir**. Amaç da aynıdır, beklenen indirimli ödülü en çoklayan politikayı bulmak:
$$\pi^*=\arg\max_\pi\sum_{t=0}^h\gamma^tE_\pi[r_t].$$
(Kaynakta $\arg\max$ eylem $a$ üzerinden yazılmış; en çoklama politikalar üzerindendir.) Model bilinmediği için ajan ortamla **etkileşerek** öğrenir. Eylem seçer, ödülü ve sonraki durumu gözler, deneyimden çıkarım yapar.

Bisiklet sürmek iyi bir örnektir. Bisikletin dinamiği (açısal momentum, sürtünme, ağırlık merkezi) çoğu insanın hesaplayabileceğinin çok ötesindedir. Yine de neredeyse herkes fizik bilmeden bisiklet sürmeyi öğrenir: deneyerek, düşerek, düzelterek. RL'nin temel fikri budur.

### Ajanın bileşenleri ve türleri {#ajan}

Bir RL ajanı aşağıdaki bileşenlerden bazılarını içerebilir:

1. **Model:** ortamın dinamiği ve ödülü. Geçiş modeli $P(s_{t+1}\mid s_t,a_t)$ ve ödül modeli $R(s_t,a_t)$ (ya da rastgele ödüllerde $P(r_t\mid s_t,a_t)$).
2. **Politika:** ajanın eylem seçimi, $\pi(s_t)$.
3. **Değer fonksiyonu:** politikanın beklenen toplam ödülü, $V(s_t)$.

Bileşenlere göre ajanlar iki eksende sınıflanır:

| sınıf | politika | değer fonksiyonu | örnek |
|---|---|---|---|
| değer tabanlı | örtük (değerden açgözlü) | var | Q-öğrenme, DQN (M11.5–6) |
| politika tabanlı | açık | yok | REINFORCE (M11.9) |
| aktör–eleştirmen | açık (aktör) | var (eleştirmen) | A2C, PPO (M11.9, M11.12) |

| sınıf | model | örnek |
|---|---|---|
| modele dayalı | geçiş ve ödül açıkça modellenir | Dyna, MCTS, AlphaZero (M11.10–11) |
| modelsiz | model yok (örtük) | MC, TD, Q-öğrenme (M11.4–6) |

Bu modül modelsiz ve değer tabanlı yöntemlerin ilk adımını atar: sabit bir $\pi$ politikasının değerini model olmadan **kestirmek** (politika değerlendirme ya da tahmin problemi). En iyi politikayı bulmak (kontrol problemi) M11.5'tedir.

### Monte Carlo değerlendirme {#mc}

Değerin tanımı bir beklentidir: $V^\pi(s)=E_\pi\bigl[\sum_t\gamma^tr_t\mid s_0=s\bigr]$. Model bilinmese de beklenti örneklerle kestirilebilir. $s$'den başlayan (ya da $s$'den geçen) tam bölümler oynanır, her birinin getirisi hesaplanır ve ortalaması alınır:
$$V^\pi(s)\approx\frac1{n(s)}\sum_{k=1}^{n(s)}G_k,\qquad G_k=\sum_{t\ge t_k}\gamma^{t-t_k}r_t^{(k)}.$$
$t_k$, $k$. bölümde $s$'nin ziyaret edildiği adımdır. Getiri o andan itibaren indirimlenir. (Kaynaktaki formül indirimi $\gamma^t$ ile mutlak zamandan başlatıyor; bu yalnız $s$ bölümün ilk durumuysa doğrudur.) Bir bölümde $s$ birden çok kez ziyaret edilebilir. **İlk ziyaret MC** yalnız ilk ziyaretin getirisini kullanır ve örnekler bağımsızdır. **Her ziyaret MC** hepsini kullanır; örnekler birbirine bağımlıdır ama yine tutarlıdır. MC'nin iki koşulu vardır: bölümler biter (epizodik görev) ve öğrenme ancak bölüm sonunda yapılabilir.

:::türetim Artımlı ortalama ve öğrenme oranı
Ortalamayı her seferinde baştan hesaplamak gerekmez. $V_n$, ilk $n$ getirinin ortalaması olsun:
$$V_n=\frac1n\sum_{k=1}^nG_k=\frac1n\Bigl(G_n+(n-1)V_{n-1}\Bigr)=V_{n-1}+\frac1n\bigl(G_n-V_{n-1}\bigr).$$
Bu, öğrenmenin genel kalıbıdır:
$$\text{yeni tahmin}\leftarrow\text{eski tahmin}+\alpha\,(\text{hedef}-\text{eski tahmin}).$$
$\alpha=1/n(s)$ tam örnek ortalamasıdır. $n(s)$ durum başına bir sayaçtır. Sabit bir $\alpha$ ise üstel ağırlıklı bir ortalamadır: $k$ adım önceki gözlem $(1-\alpha)^k$ ağırlık alır. Ortam zamanla değişiyorsa (durağan değilse) sabit $\alpha$ tercih edilir.
:::

### Zamansal fark (TD) değerlendirme {#td}

MC bölüm sonunu bekler. Bellman denklemi daha kısa bir yol gösterir:
$$V^\pi(s)=E_\pi\Bigl[r\mid s,\pi(s)\Bigr]+\gamma\sum_{s^+}P(s^+\mid s,\pi(s))\,V^\pi(s^+).$$
Ödül modeli olmadığı için ilk beklenti, geçiş modeli olmadığı için ikinci toplam hesaplanamaz. Ama ikisi de **tek bir adım** oynanarak örneklenebilir. $s$'de $\pi(s)$ uygulanır, ödül $r$ ve sonraki durum $s^+$ gözlenir, ve $r+\gamma V^\pi(s^+)$ hedef olarak kullanılır. Artımlı kalıba konunca **TD(0)** güncellemesi çıkar:
$$V(s)\leftarrow V(s)+\alpha\bigl[\underbrace{r+\gamma V(s^+)-V(s)}_{\delta:\ \text{TD hatası}}\bigr].$$
Hedef, $V$'nin kendi tahminini içerir. Bir tahmin başka bir tahminle güncellenir; buna **bootstrapping** denir. TD her adımda öğrenir, bölüm sonunu beklemez ve bitmeyen görevlerde de çalışır.

```text
TD-değerlendirme(π):
    her s için V(s) ← 0,  n(s) ← 0
    s ← başlangıç durumu
    tekrarla:
        a ← π(s);  uygula;  r ve s⁺'yı gözle
        n(s) ← n(s) + 1;   α ← 1 / n(s)
        V(s) ← V(s) + α [ r + γ V(s⁺) − V(s) ]
        s ← s⁺   (s⁺ son durumsa yeni bölüme başla)
    V yakınsayana dek
```

:::türetim Robbins–Monro koşulları
Gürültülü hedeflerle yapılan artımlı güncellemelerin doğru değere yakınsaması için öğrenme oranı iki koşulu sağlamalıdır:
$$\sum_n\alpha_n=\infty,\qquad\sum_n\alpha_n^2<\infty.$$
Birinci koşul adımların toplamının sınırsız olmasını ister. Böylece başlangıç değeri ne kadar uzakta olursa olsun güncellemeler oraya ulaşabilir. İkinci koşul gürültünün etkisinin sönmesini ister. Güncellemelerin toplam varyansı $\sigma^2\sum\alpha_n^2$ ile orantılıdır ve sonlu kalmalıdır. $\alpha_n=1/n$ ikisini de sağlar ($\sum1/n$ ıraksar, $\sum1/n^2=\pi^2/6$). $\alpha_n=1/n^p$, $0.5<p\le1$ için de sağlanır. Sabit $\alpha$ ikinci koşulu bozar. Tahmin yakınsamaz, gerçek değer etrafında $\alpha$ ile büyüyen bir gürültüyle dolaşmaya devam eder. Her durum için $n$ ayrı sayılır. TD'nin yakınsaması için ayrıca her durumun sonsuz kez ziyaret edilmesi gerekir.
:::

### Yanlılık ve varyans {#yanlilik}

MC hedefi $G_t$, $V^\pi(s)$'nin **yansız** bir örneğidir. Beklentisi tam olarak değerdir. Ama bölümün bütün rastgeleliğini taşır: her eylem, her geçiş, her ödül. Varyansı yüksektir. TD hedefi $r+\gamma V(s^+)$ yalnız bir adımın rastgeleliğini taşır, varyansı düşüktür. Ama $V(s^+)$ bir tahmindir. Tahmin yanlış olduğu sürece hedef **yanlıdır**. Yanlılık öğrenme ilerledikçe azalır ve Robbins–Monro koşullarıyla TD de doğru değere yakınsar. Yani "yanlı" sonlu örneklemdeki bir özelliktir, limitteki değil.

Sutton'ın beş durumlu rastgele yürüyüşü bunu sayısal olarak gösterir. Durumlar A–E'dir ve C'den başlanır. Her adımda yarı yarıya sola ya da sağa gidilir. Soldan çıkış ödül 0, sağdan çıkış ödül 1 verir, $\gamma=1$'dir. Gerçek değerler $1/6,\dots,5/6$'dır. C'de MC hedefi 0 ya da 1'dir (varyans 1/4). TD hedefi, gerçek değerlerle, $V(B)=1/3$ ya da $V(D)=2/3$'tür (varyans 1/36). İkisinin beklentisi 0.5'tir; varyansları 9 kat farklıdır (Şekil 1).

<figure class="fig wide"><svg viewBox="0 0 640 230" role="img" aria-label="Beş durumlu rastgele yürüyüşte C durumundan Monte Carlo ve TD hedeflerinin dağılımları"><path d="M70,200.0 L370,200.0" class="mute"/><text x="64" y="204.0" text-anchor="end" class="sm">0.0</text><path d="M70,143.33333333333331 L370,143.33333333333331" class="mute"/><text x="64" y="147.33333333333331" text-anchor="end" class="sm">0.2</text><path d="M70,86.66666666666666 L370,86.66666666666666" class="mute"/><text x="64" y="90.66666666666666" text-anchor="end" class="sm">0.4</text><path d="M70,30.0 L370,30.0" class="mute"/><text x="64" y="34.0" text-anchor="end" class="sm">0.6</text><text x="94" y="216" text-anchor="middle" class="sm">0</text><text x="178.0" y="216" text-anchor="middle" class="sm">1/3</text><text x="220.0" y="216" text-anchor="middle" class="sm">0.5</text><text x="262.0" y="216" text-anchor="middle" class="sm">2/3</text><text x="346" y="216" text-anchor="middle" class="sm">1</text><rect x="77" y="59.73583333333332" width="16" height="140.26416666666668" class="ca-f"/><rect x="329" y="56.93083333333334" width="16" height="143.06916666666666" class="ca-f"/><rect x="178.99991599999998" y="57.99333333333334" width="16" height="142.00666666666666" class="cb-f"/><rect x="263.000084" y="58.67333333333332" width="16" height="141.32666666666668" class="cb-f"/><path d="M220.0,30 L220.0,200" stroke="currentColor" stroke-dasharray="4 3"/><text x="224.0" y="40" class="sm">V(C) = 0.5</text><text x="70" y="18" class="sm">C durumundan hedeflerin dağılımı (20 000 bölüm)</text><text x="410" y="40" class="sm">Mavi: MC hedefi (getiri)</text><text x="410" y="59" class="sm">  ort. 0.505, var. 0.2500 (1/4)</text><text x="410" y="78" class="sm">Turuncu: TD hedefi r + V(s')</text><text x="410" y="97" class="sm">  ort. 0.500, var. 0.0278 (1/36)</text><text x="410" y="116" class="sm"></text><text x="410" y="135" class="sm">Varyans oranı ≈ 9.0: aynı doğrulukta</text><text x="410" y="154" class="sm">TD hedefi çok daha az örnek ister.</text><text x="410" y="173" class="sm">Ama TD'nin hedefi V tahminini kullanır:</text><text x="410" y="192" class="sm">tahmin yanlışsa hedef de yanlıdır.</text></svg><figcaption>Şekil 1 — Aynı beklentinin iki kestiricisi. MC hedefi bölümün bütün rastgeleliğini taşır (0 ya da 1); TD hedefi yalnız bir adımın rastgeleliğini taşır (1/3 ya da 2/3). Gerçek değerlerle ikisi de yansızdır, varyansları 9 kat farklıdır (simüle edildi).</figcaption></figure>

| | Monte Carlo | TD(0) |
|---|---|---|
| hedef | $G_t$ (tam getiri) | $r+\gamma V(s^+)$ |
| yanlılık | yansız | tahmin doğru olana dek yanlı |
| varyans | yüksek | düşük |
| ne zaman öğrenir | bölüm sonunda | her adımda |
| bitmeyen görev | uygun değil | uygun |
| Markov varsayımı | gerekmez | kullanır |

Kaynağın karşılaştırması TD için "daha az yörünge gerekir" diyor. Bu çoğu Markov problemde doğrudur, ama genel bir kural değildir. Süreç Markov değilse ya da değer fonksiyonu yaklaşık temsil ediliyorsa (M11.6), TD'nin yanlılığı kalıcı olabilir ve MC daha iyi sonuç verebilir. Aşağıdaki deneyin ön ayarlarında rastgele yürüyüşte TD'nin hatası MC'ninkinin altında kalıyor; öğrenme oranlarını değiştirerek sınırları denenebilir.

::lab l1

### Toplu güncelleme: MC ve TD farklı cevaplara yakınsar {#toplu}

Deneyim sınırlıysa aynı veri tekrar tekrar kullanılabilir. Güncellemeler bütün veri üzerinde toplanır ve yakınsayana dek tekrarlanır (toplu ya da batch güncelleme). Sonsuz veriyle iki yöntem de aynı değere yakınsar. Sonlu veride ise **farklı** sabit noktalara gider.

Sutton'ın örneği sekiz bölümdür. Birinde A'dan başlanır, 0 ödülle B'ye geçilir, B'den 0 ödülle bölüm biter. Kalan yedisi B'de başlayıp hemen biter; altısında ödül 1, birinde 0'dır.

- **B'nin değerinde** iki yöntem anlaşır. B sekiz kez gözlendi ve altısında getiri 1'dir: $V(B)=6/8=0.75$.
- **A'nın değerinde** ayrılırlar. Toplu MC, A'nın tek gözlenen getirisini (0) ortalar: $V(A)=0$. Toplu TD, A'dan B'ye 0 ödülle geçildiğini öğrenir ve $V(A)=0+V(B)=0.75$ verir.

Hangisi doğru? Toplu MC, **eğitim verisindeki** getirilere ortalama kare hatayı en küçükleyen değerleri bulur. Toplu TD ise veriden en çok olabilirlikle kestirilen Markov modelinin (geçiş oranları ve ortalama ödüller; M11.1'deki sayım oranları) tam değerini bulur. Buna **kesinlik eşdeğeri** kestirimi denir: model kesinmiş gibi davranılır. Süreç gerçekten Markov ise TD'nin cevabı gelecekteki veride daha az hata yapar. A'dan sonra B'nin geldiği ve B'nin değerinin 0.75 olduğu bilgisi, tek bir şanssız bölümden daha güvenilirdir. Rastgele yürüyüşte toplu TD neredeyse her veri miktarında toplu MC'den daha düşük hata verir.

::lab l2

### n adımlı getiriler ve TD(λ) {#td-lambda}

MC ve TD(0) bir yelpazenin iki ucudur. Arada **n adımlı getiriler** vardır: $n$ gerçek ödül toplanır, sonra tahmine geçilir.
$$G_t^{(n)}=r_t+\gamma r_{t+1}+\dots+\gamma^{n-1}r_{t+n-1}+\gamma^nV(s_{t+n}).$$
$n=1$ TD(0)'ın hedefidir. $n=\infty$ (bölüm sonuna kadar) MC'nin hedefidir. $n$ büyüdükçe yanlılık azalır, varyans artar. Sorunun ölçeğine göre ara bir $n$ çoğu zaman ikisinden de iyi sonuç verir.

**λ-getirisi** bütün $n$'leri geometrik ağırlıklarla birleştirir:
$$G_t^\lambda=(1-\lambda)\sum_{n=1}^\infty\lambda^{n-1}G_t^{(n)}.$$
Ağırlıkların toplamı $(1-\lambda)\sum_n\lambda^{n-1}=1$'dir. $\lambda=0$ TD(0)'dır, $\lambda=1$ MC'dir (Şekil 2).

<figure class="fig wide"><svg viewBox="0 0 640 240" role="img" aria-label="Lambda getirisinde n adımlık getirilerin geometrik ağırlıkları, üç lambda değeri için"><path d="M60,198.0 L440,198.0" class="mute"/><text x="54" y="202.0" text-anchor="end" class="sm">0.0</text><path d="M60,164.0 L440,164.0" class="mute"/><text x="54" y="168.0" text-anchor="end" class="sm">0.1</text><path d="M60,130.0 L440,130.0" class="mute"/><text x="54" y="134.0" text-anchor="end" class="sm">0.2</text><path d="M60,96.0 L440,96.0" class="mute"/><text x="54" y="100.0" text-anchor="end" class="sm">0.3</text><path d="M60,62.0 L440,62.0" class="mute"/><text x="54" y="66.0" text-anchor="end" class="sm">0.4</text><path d="M60,28.0 L440,28.0" class="mute"/><text x="54" y="32.0" text-anchor="end" class="sm">0.5</text><text x="75.83333333333333" y="214" text-anchor="middle" class="sm">1</text><rect x="63.0" y="28.0" width="7.6" height="170.0" class="ca-f"/><rect x="71.6" y="130.0" width="7.6" height="68.0" class="cb-f"/><rect x="80.1" y="181.0" width="7.6" height="17.0" class="mute-f"/><text x="107.5" y="214" text-anchor="middle" class="sm">2</text><rect x="94.7" y="113.0" width="7.6" height="85.0" class="ca-f"/><rect x="103.2" y="143.6" width="7.6" height="54.4" class="cb-f"/><rect x="111.8" y="181.8" width="7.6" height="16.2" class="mute-f"/><text x="139.16666666666669" y="214" text-anchor="middle" class="sm">3</text><rect x="126.3" y="155.5" width="7.6" height="42.5" class="ca-f"/><rect x="134.9" y="154.5" width="7.6" height="43.5" class="cb-f"/><rect x="143.4" y="182.7" width="7.6" height="15.3" class="mute-f"/><text x="170.83333333333334" y="214" text-anchor="middle" class="sm">4</text><rect x="158.0" y="176.8" width="7.6" height="21.2" class="ca-f"/><rect x="166.6" y="163.2" width="7.6" height="34.8" class="cb-f"/><rect x="175.1" y="183.4" width="7.6" height="14.6" class="mute-f"/><text x="202.5" y="214" text-anchor="middle" class="sm">5</text><rect x="189.7" y="187.4" width="7.6" height="10.6" class="ca-f"/><rect x="198.2" y="170.1" width="7.6" height="27.9" class="cb-f"/><rect x="206.8" y="184.2" width="7.6" height="13.8" class="mute-f"/><text x="234.16666666666669" y="214" text-anchor="middle" class="sm">6</text><rect x="221.3" y="192.7" width="7.6" height="5.3" class="ca-f"/><rect x="229.9" y="175.7" width="7.6" height="22.3" class="cb-f"/><rect x="238.4" y="184.8" width="7.6" height="13.2" class="mute-f"/><text x="265.83333333333337" y="214" text-anchor="middle" class="sm">7</text><rect x="253.0" y="195.3" width="7.6" height="2.7" class="ca-f"/><rect x="261.6" y="180.2" width="7.6" height="17.8" class="cb-f"/><rect x="270.1" y="185.5" width="7.6" height="12.5" class="mute-f"/><text x="297.5" y="214" text-anchor="middle" class="sm">8</text><rect x="284.7" y="196.7" width="7.6" height="1.3" class="ca-f"/><rect x="293.2" y="183.7" width="7.6" height="14.3" class="cb-f"/><rect x="301.8" y="186.1" width="7.6" height="11.9" class="mute-f"/><text x="329.1666666666667" y="214" text-anchor="middle" class="sm">9</text><rect x="316.3" y="197.3" width="7.6" height="0.7" class="ca-f"/><rect x="324.9" y="186.6" width="7.6" height="11.4" class="cb-f"/><rect x="333.4" y="186.7" width="7.6" height="11.3" class="mute-f"/><text x="360.83333333333337" y="214" text-anchor="middle" class="sm">10</text><rect x="348.0" y="197.7" width="7.6" height="0.3" class="ca-f"/><rect x="356.6" y="188.9" width="7.6" height="9.1" class="cb-f"/><rect x="365.1" y="187.3" width="7.6" height="10.7" class="mute-f"/><text x="392.5" y="214" text-anchor="middle" class="sm">11</text><rect x="379.7" y="197.8" width="7.6" height="0.2" class="ca-f"/><rect x="388.2" y="190.7" width="7.6" height="7.3" class="cb-f"/><rect x="396.8" y="187.8" width="7.6" height="10.2" class="mute-f"/><text x="424.1666666666667" y="214" text-anchor="middle" class="sm">12</text><rect x="411.3" y="197.9" width="7.6" height="0.1" class="ca-f"/><rect x="419.9" y="192.2" width="7.6" height="5.8" class="cb-f"/><rect x="428.4" y="188.3" width="7.6" height="9.7" class="mute-f"/><text x="250.0" y="230" text-anchor="middle" class="sm">n (adım sayısı)</text><text x="60" y="16" class="sm">n adımlık getirinin λ-getirisindeki ağırlığı (1 − λ)λ<tspan baseline-shift="4" font-size="75%">n−1</tspan></text><text x="470" y="40" class="sm">λ = 0.5: yarısı ilk 1 adımda,</text><text x="470" y="57" class="sm">  ortalama ufuk 1/(1 − λ) = 2</text><text x="470" y="74" class="sm">λ = 0.8: yarısı ilk 4 adımda,</text><text x="470" y="91" class="sm">  ortalama ufuk 1/(1 − λ) = 5</text><text x="470" y="108" class="sm">λ = 0.95: yarısı ilk 14 adımda,</text><text x="470" y="125" class="sm">  ortalama ufuk 1/(1 − λ) = 20</text><text x="470" y="142" class="sm">λ = 0: TD(0)</text><text x="470" y="159" class="sm">λ = 1: Monte Carlo</text><rect x="470" y="204" width="10" height="10" class="ca-f"/><text x="484" y="213" class="sm">λ = 0.5</text><rect x="470" y="220" width="10" height="10" class="cb-f"/><text x="484" y="229" class="sm">λ = 0.8</text><rect x="550" y="204" width="10" height="10" class="mute-f"/><text x="564" y="213" class="sm">λ = 0.95</text></svg><figcaption>Şekil 2 — λ-getirisi bütün n adımlık getirilerin geometrik ağırlıklı ortalamasıdır; ağırlıkların toplamı 1'dir. λ büyüdükçe uzak getirilerin ağırlığı artar ve tahmin MC'ye yaklaşır (hesaplandı).</figcaption></figure>

:::türetim İleri bakış ve geri bakış: uygunluk izleri
λ-getirisini doğrudan hesaplamak geleceği beklemeyi gerektirir (ileri bakış). **Uygunluk izi** aynı güncellemeyi geçmişe dağıtarak yapar (geri bakış). Her durum için bir iz $e(s)$ tutulur. Her adımda bütün izler $\gamma\lambda$ ile söner ve ziyaret edilen durumunki 1 artar:
$$e(s)\leftarrow\gamma\lambda\,e(s)+\mathbb 1[s=s_t],\qquad V(s)\leftarrow V(s)+\alpha\,\delta_t\,e(s)\quad\forall s.$$
Her TD hatası, son ziyaret edilen durumlara, ne kadar yakın zamanda ve ne sıklıkta ziyaret edildiklerine göre paylaştırılır. İki bakışın eşdeğerliği şu özdeşlikten gelir:
$$G_t^\lambda-V(s_t)=\sum_{k=t}^\infty(\gamma\lambda)^{k-t}\delta_k.$$
Bunu görmek için $G_t^{(n)}-V(s_t)=\sum_{k=t}^{t+n-1}\gamma^{k-t}\delta_k$ yazılır; ara $V$ terimleri teleskopik olarak sadeleşir. Sonra $(1-\lambda)\lambda^{n-1}$ ağırlıklarıyla toplanır; $\delta_k$'nın katsayısı $\gamma^{k-t}\sum_{n>k-t}(1-\lambda)\lambda^{n-1}=(\gamma\lambda)^{k-t}$ olur. Değerler bölüm içinde güncellenmezse (çevrimdışı), bir bölümdeki toplam geri bakış güncellemesi ileri bakışınkine eşittir. Bu modülün d aşaması bunu PyTorch'ta sayısal olarak doğruluyor.
:::

### Özet {#ozet}

- RL'de model bilinmez; ajan etkileşerek öğrenir. Ajanlar model, politika ve değer fonksiyonundan hangilerini taşıdıklarına göre sınıflanır: değer tabanlı, politika tabanlı, aktör–eleştirmen; modele dayalı, modelsiz.
- MC değerlendirme tam bölümlerin getirilerini ortalar; ilk ziyaret ya da her ziyaret biçimleri vardır. Getiri ziyaret anından itibaren indirimlenir.
- Artımlı ortalama: yeni = eski + α(hedef − eski). α = 1/n örnek ortalamasıdır, sabit α üstel ağırlıklıdır.
- TD(0): $V(s)\leftarrow V(s)+\alpha[r+\gamma V(s^+)-V(s)]$. Tek adımın deneyimiyle ve bootstrapping ile öğrenir.
- Robbins–Monro: $\sum\alpha=\infty$, $\sum\alpha^2<\infty$; sabit α ile tahmin gürültülü kalır.
- MC yansız ve yüksek varyanslıdır; TD tahmin doğru olana dek yanlı ve düşük varyanslıdır. Rastgele yürüyüşte C'deki hedeflerin varyansları 1/4'e karşı 1/36'dır.
- Toplu güncellemede MC eğitim hatasını en küçükler, TD kesinlik eşdeğeri kestirimini bulur (A–B örneği: V(A) = 0'a karşı 0.75).
- n adımlı getiriler ve λ-getirisi iki uç arasında yanlılık–varyans dengesi kurar. Uygunluk izleri λ-getirisini geri bakışla, adım adım uygular.
