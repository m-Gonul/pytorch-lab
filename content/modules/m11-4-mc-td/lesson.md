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
| yanlılık | ilk-ziyaret ortalaması uygun koşullarla yansız; her-ziyaret sonlu veride yanlı olabilir | tahmin doğru olana dek yanlı |
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

### Tam bölümden öğrenmek: kaynak Bölüm 5 {#sutton-mc-temel}

Sutton ve Barto (2018), s.91–117 içindeki Monte Carlo (MC) paketinin değerlendirme kısmı burada, kontrol kısmı [M11.5](m11-5-sarsa-q.html#sutton-mc-kontrol) içindedir. Kaynak gösteriminde $R_{t+1}$, eylem $A_t$ sonrasındaki ödüldür; önceki kodlarda `r[t]` aynı ödülü tutar. Kaynak Bölüm 5.8–5.9 ileri ayrıntıdır; temel ilk/her ziyaret anlatımından sonra okunabilir.

Model bir geçişin bütün olasılıklarını söylemeden de **örnek bir bölüm üretebilir**. MC gerçek deneyimde veya böyle bir benzeticide, tamamlanmış bölümlerin getirilerini ortalar. Bu yüzden tam geçiş/ödül modeli gerekmez; yalnız örnekleme modeli bile işe yarar. Kaynak bölüm başında daha güçlü bir epizodik varsayım koyar: hangi eylemler seçilirse seçilsin bütün bölümler sonunda sonlanır. Yöntemlerin burada kullanılan tahmin güvenceleri en azından örneklenen politikalar altında sonlanma ve gerekli integrallenebilirliği ister. Yarış pistinde başlangıçta sonsuz bekleme gibi yasal davranışlar güçlü varsayıma uymaz; uygulama sonlanan keşif davranışını ve güvenlik bütçesini ayrıca açıklar. MC güncellemesi bölüm bitince yapılır; bölümler arasında artımlıdır, her çevre adımında güncelleme yapan TD ile aynı çevrimiçi zamanlama değildir.

Tek bir durumda her eylemi denemek, bağlamsal bandit gibi görünebilir. Fakat eylemin getirisi sonraki durumlarda izlenen politikaya bağlıdır. O politika değişince daha önceki eylemlerin getiri dağılımı da değişir: bunlar birbirine bağlı, durağan olmayan alt problemlerdir. Çözüm yine GPI, yani değerlendirme ile iyileştirmenin etkileşimidir. DP tam model üzerinden beklenti hesaplarken MC örnek getirileri kullanır.

### İlk ziyaret, her ziyaret ve önyüklemesiz backup {#sutton-mc-ziyaret}

Sabit bir $\pi$ için hedef $v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]$. Bir bölümde $s$ birkaç kez görülebilir. **İlk ziyaret** yalnız zaman sırasındaki ilk gelişin getirisini; **her ziyaret** bütün gelişlerin getirilerini kullanır. Geriye giderken ilk ziyaret arıyorsanız, durumu daha önceki `states[:t]` içinde arayın. Geri taramada ilk rastladığınız durum, zaman sırasındaki **son** ziyarettir.

Kaynak s.92 ilk-ziyaret sözde-kodunun aynı işi yapan artımlı karşılığı:

```python
V, count = {}, {}
for states, rewards in completed_episodes(policy):
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        s = states[t]
        if s in states[:t]:
            continue
        count[s] = count.get(s, 0) + 1
        old = V.get(s, 0.0)
        V[s] = old + (G - old) / count[s]
```

Başlangıç $V$ keyfî, sayımlar 0 olabilir. Kaynak getirilerin listesini tutup ortalar; aynı sonuca liste tutmadan ulaşılır (Alıştırma 5.9). Her ziyaret için `if` koşulu kaldırılır. İlk-ziyaret biçiminin kullanımı 1940 yıllarına uzanır. Fonksiyon yaklaşımı ve uygunluk izleriyle her-ziyaret biçimi daha doğal birleşir; Bölüm 9/12 bu bağlantıyı genişletir.

:::türetim Hangi anlamda yansız, hangi hızda yakınsar?
Sabit politika ve bağımsız bölümler altında, bir duruma ulaşmış bölümlerin ilk ziyaret getirileri aynı koşullu dağılımdan gelir. $n$ böyle getirinin ortalaması yansızdır; sonlu varyans $\sigma^2$ varsa ortalamanın varyansı $\sigma^2/n$, standart hatası $\sigma/\sqrt n$. Her-ziyaret getirileri aynı bölüm içinde bağımlıdır, ziyaret sayısı da rastgeledir. Bu yüzden sonlu veri oran kestiricisi yanlı olabilir; uygun sonlanma/moment koşullarıyla tutarlıdır. Kaynağın “quadratically” sözü burada **ortalama karede yakınsamayı** anlatır; her yinelemede hata karesi kadar azalma iddiası değildir.

Alıştırma 5.5: tek durum, tek eylem, öz döngü olasılığı $p$, terminal olasılığı $1-p$, her geçişte+1 ve $\gamma=1$. Gözlenen tek bölüm 10 adım ise ilk ziyaret 10; her ziyaret $(10+9+\cdots+1)/10=5.5$ verir. Gerçek değer $1/(1-p)$; bir bölümün cevabı gerçek değerle karıştırılmaz.
:::

<figure class="fig wide"><svg viewBox="0 0 640 390" role="img" aria-label="MC backup: durum ve eylem kökleri, tam tek yol" xmlns="http://www.w3.org/2000/svg"><defs><marker id="MC-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><text x="160" y="27" text-anchor="middle" style="font-size:16px;fill:var(--ink)">V: kök s</text><circle cx="160" cy="60" r="7" fill="var(--paper)" stroke="var(--ink)"/><path d="M160,67 L160,80" fill="none" stroke="var(--ink)" stroke-width="1.5"/><circle cx="160" cy="84" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M160,89 L160,130" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="218" y="100" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R1=1</text><circle cx="160" cy="138" r="7" fill="var(--paper)" stroke="var(--ink)"/><path d="M160,145 L160,158" fill="none" stroke="var(--ink)" stroke-width="1.5"/><circle cx="160" cy="162" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M160,167 L160,208" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="218" y="178" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R2=-2</text><circle cx="160" cy="216" r="7" fill="var(--paper)" stroke="var(--ink)"/><path d="M160,223 L160,236" fill="none" stroke="var(--ink)" stroke-width="1.5"/><circle cx="160" cy="240" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M160,245 L160,286" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="218" y="256" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R3=4</text><rect x="152" y="286" width="16" height="16" fill="var(--paper)" stroke="var(--ink)"/><text x="160" y="324" text-anchor="middle" style="font-size:16px;fill:var(--ink)">terminal</text><text x="160" y="361" text-anchor="middle" style="font-size:16px;fill:var(--ink)">γ=0.9; G₀=2.44</text><text x="475" y="27" text-anchor="middle" style="font-size:16px;fill:var(--ink)">Q: kök (s,a)</text><circle cx="475" cy="60" r="7" fill="var(--ink)" stroke="var(--ink)"/><path d="M475,65 L475,130" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="533" y="100" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R1=1</text><circle cx="475" cy="138" r="7" fill="var(--paper)" stroke="var(--ink)"/><path d="M475,145 L475,158" fill="none" stroke="var(--ink)" stroke-width="1.5"/><circle cx="475" cy="162" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M475,143 L475,208" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="533" y="178" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R2=-2</text><circle cx="475" cy="216" r="7" fill="var(--paper)" stroke="var(--ink)"/><path d="M475,223 L475,236" fill="none" stroke="var(--ink)" stroke-width="1.5"/><circle cx="475" cy="240" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M475,221 L475,286" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="533" y="256" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R3=4</text><rect x="467" y="286" width="16" height="16" fill="var(--paper)" stroke="var(--ink)"/><text x="475" y="324" text-anchor="middle" style="font-size:16px;fill:var(--ink)">terminal</text><text x="475" y="361" text-anchor="middle" style="font-size:16px;fill:var(--ink)">γ=0.9; G₀=2.44</text></svg><figcaption>Şekil 3 — MC-V ve Alıştırma5.3 MC-Q backup. Açık durum/siyah eylem/kare terminal; tek örnek yolun üç ödülüyle G₀ hesaplandı. Q için kök eylem zaten koşulludur. Kaynak s.95 numarasız V şemasının ilişkileri yeniden türetildi.</figcaption></figure>

MC backup, tek örnek yolunu terminale kadar izler; ara düğümlerde **başka $V$ kestirimi kullanmaz**. Tam dallanan DP beklenti ağacı değildir. $q_\pi(s,a)$ için de kökte eylem zorlanır, ardından politika izlenir; terminale kadar bütün ödüller alınır (Alıştırma 5.3). Bir durumun hesabı başka durumun kestirimine bağlı olmadan yapılabilir; önemli birkaç başlangıca hesap bütçesi ayırabilirsiniz. Bu, kestirimlerin istatistiksel bağımsızlığı değildir: $A\to B\to$ terminal yolunda aynı Bernoulli son ödülünü kullanırsanız $\hat V(A)=\hat V(B)$, tek bölümde kovaryans 1/4 olur. Durum uzayının bütününü listeleme gereği yoktur; bölüm uzunluğu ve örnek üretme maliyeti problem boyutuyla yine değişebilir.

### Blackjack: 200 durum, doğal 21 ve dört yüzey {#sutton-blackjack}

Kaynak Örnek 5.1: amaç 21 değerini aşmadan krupiyeden yüksek toplamdır. Eylemler çek/dur; resimli kart 10, as toplamı aşırmıyorsa 11, aksi halde 1 sayılır. Kazanç+1, beraberlik 0, kayıp−1; $\gamma=1$, bütün ara ödüller 0. Sonsuz deste yerine koyarak çekmeye denktir: A…9 olasılığı 1/13,10 olasılığı 4/13. Krupiye iki karttan birini açık gösterir,17 altındayken çeker,17 ve üzerinde durur; kullanılabilir aslı 17 de durur. Oyuncu aşarsa krupiyeye bakılmadan kaybeder.

Başlangıçtaki iki kart as+10 ise **doğal 21** vardır: krupiyede de doğal 21 varsa beraberlik, yoksa karar almadan kazanılır. Doğal olmayan oyunlarda 12 altındaki toplamlar zorunlu çekilerek karar aralığına gelir. Durum üçlüsü `(oyuncu toplamı12…21, krupiye açık kartıA…10, kullanılabilir as var/yok)`: $10\times10\times2=200$. Kullanılabilir as, o anda 11 sayılan astır; yalnız elde as bulunması yeterli değildir. Deste yerine koyulmasa kalan kartlar da bilgi taşır, bu 200 durumun Markov modeli yetmezdi.

Değerlendirilen politika 20/21 toplamında durur, diğer karar toplamlarında çeker. Şekil 5.1 kaynak koşulu 10000 ve 500000 bölümdür; her bütçede aslı/asız iki yüzey, oyuncu toplamı 12–21, krupiyeA–10 ve değer−1…+1 eksenleriyle gösterilir. Nadir kullanılabilir-as durumları daha az örnek alır. Tam kuralları bilsek de DP için krupiyenin bütün bitiş dağılımlarını çıkarmak gerekir: örneğin oyuncu 14 toplamında durunca+1 olasılığı, açık karta göre krupiyenin aşma olasılığıdır. Bu çok adımlı olasılık hesabı karmaşık ve hataya açıktır; bir örnek eli üretmek çok daha kolay olabilir. Örnek üretici ile tam açık model arasındaki fark burada önemlidir. İlk/her ziyaret bu oyunda aynı veridir: aynı kullanılabilir-as sınıfında toplam artar; as 11 sayılmaktan 1 sayılmaya geçince sınıf değişir ve geri dönmez. Böylece **tam üçlü bir elde tekrar edemez** (Alıştırma 5.2).

:::uyarı Kaynakla uyum düzeltmesi: eski x 1 içindeki doğal kuralı
Eski x 1 yalnız son toplamları karşılaştırıyordu. Başlangıçtaki doğal 21 şimdi krupiye sonradan 21 yapmadan önce+1/0 ile çözülür. `natural_rule=False` önceki karşılaştırma çeşidini denemeye izin verir; varsayılan kaynak kuralıdır. ES ile zorlanan 21 ise doğal olmayan bir karar durumu olarak yorumlanır: örneğin üç kartla elde edilebilir. Böylece doğal oyun sonlanması ile 200 karar durumunu karıştırmayız.
:::

::lab l 3

Alıştırma 5.1 yüzey yorumları:20/21 yükselişi yalnız “büyük toplam iyidir” değildir; politika tam burada çekmekten durmaya geçer. Krupiye ası güçlü olduğu için A sütunu çoğunlukla düşüktür. Küçük oyuncu toplamında kullanılabilir as koruma sağlar:10 çekince toplamdan 10 çıkarılarak aşma önlenebilir. Tam model çözümüyle eğri gürültüsünü politika etkisinden ayırın. Python 500000 bölüm RMS farkı 0.0286822, JS 0.0261107 ölçüldü; farklı RNG kullanmaları aynı elleri üretmez.

### Tel yüzeyinde tek bir yüksekliği bulmak {#sutton-dirichlet}

Kaynak Örnek 5.2 içindeki düzensiz tel, üzerinde gerili yüzey, düzlemdeki $x,y$ ve seçilen $P$ noktasının yüksekliği $h$ şu modelle yeniden hesaplanır. İç ızgarada dört komşudan biri eşit olasılıkla seçilir; tel sınırına varınca yürüyüş biter ve o sınırın yüksekliği terminal ödülüdür. Kaynağın doğrusal denge idealizasyonunda komşulardan gelen net kuvvet sıfırdır; bu, küçük bir çevrede yüksekliğin çevredeki yüksekliklerin ortalaması olmasına götürür. Dört komşulu ızgara bu yerel ortalama ilişkisini ayrıklaştırır. Harmonik çözümde her iç nokta dört komşusunun ortalamasıdır. Kaynak fotoğrafı Hersh–Griego (1969) kökenlidir ve ©1969 Scientific American kredi notunu taşır; burada o fotoğraf kullanılmaz. DP bütün ızgarayı yineler; MC yalnız ilgilendiğiniz $P$ noktasından başlayan yürüyüşlerin son sınır yüksekliğini ortalar. Ara yükseklik kestirimlerine ihtiyaç duymaz.

::lab l 4

:::türetim Dirichlet çözümü ve fiziksel yorumun sınırı
Sınır yükseklikleri $g$ olsun. Sonlu soğurucu yürüyüşün ilk adım çözümlemesi $h(s)=\frac14\sum_{s'\sim s}h(s')$, sınırda $h=g$ verir. Tek çözüm hem ayrık Laplace denklemini hem de beklenen terminal yüksekliğini sağlar. Yürüyüşler bağımsızsa ortalama bu beklentiye yakınsar. Deneyde seçtiğimiz $h(x,y)=k(x^2-y^2+0.6xy)$ polinomu harmoniktir; simetrik ikinci farklar da tam sıfırdır. Analitik yüzeyle DP en büyük farkı varsayılan ölçek için $3\times10^{-10}$ altında ölçüldü.

Kaynak notu: genel sabun filmi **alanı** en aza indirir. $\int\sqrt{1+h_x^2+h_y^2}\,dxdy$ fonksiyonelinin Euler–Lagrange denklemi $\nabla\cdot(\nabla h/\sqrt{1+|\nabla h|^2})=0$. Küçük eğimde bu, $\Delta h=0$ olur. Tam minimal yüzey ile harmonik çözüm genel olarak aynı değildir: $h=xy$ için Laplace sıfır, minimal-yüzey denkleminin açılmış payı $-2xy$ olur. Deneyimiz fotoğrafın kopyası değil, kaynakta kullanılan harmonik Dirichlet açıklamasının hesaplanan modelidir.
:::

### Başka bir politikanın verisiyle tahmin {#sutton-is}

Keşif yapan **davranış politikası** $b$ veri toplar; **hedef politika** $\pi$ değerlendirilir. Hedef belirlenimci olabilir. Politika üstü yöntem $b=\pi$ özel durumudur. Politika dışı öğrenme daha genel, çoğu kez daha yüksek varyanslıdır; eski bir kontrolcünün veya insan uzmanın verisinden öğrenebilir. Dünya dinamikleri için çok adımlı tahminler öğrenme bağlantısı kaynak§17.2, Sutton (2009), Sutton ve arkadaşları (2011) içindedir. Burada tahmin türetiminde iki politika sabittir.

**Kapsama koşulu:** $\pi(a|s)>0\Rightarrow b(a|s)>0$. Hedefle farklı eylemler seçen davranış bu durumlarda rastgele olmalıdır. Bu koşul, başlangıç dağılımından ilgili durumlara gerçekten ulaşma ve yeterli ziyaret gereğinin yerine geçmez.

:::türetim Yol olasılıklarından önem oranı: kaynak (5.3–5.4)
$S_t$ verilince yol olasılığı $\prod_{k=t}^{T-1}\pi(A_k|S_k)p(S_{k+1}|S_k,A_k)$ olur. Ödüller rastgelse ortak geçiş/ödül çekirdeği kullanılır; yine iki politikada aynıdır. Oranı alınca bilinmeyen çevre terimleri sadeleşir:

$$\rho_{t:T-1}=\prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}.$$

Kapsama ve sonlu beklenen mutlak ağırlıklı getiri altında, davranış altında yolları toplayarak

$$\mathbb E_b[\rho_{t:T-1}G_t\mid S_t=s]=\mathbb E_\pi[G_t\mid S_t=s]=v_\pi(s).$$

Oransız $G_t$ ortalaması ise $v_b(s)$ değerini hedefler. Geçiş modelinin sayıları önem oranını hesaplamak için gerekmez.
:::

Kaynakta zaman sayacı bölümler arasında sürer: ilk terminal 100 ise sonraki başlangıç 101. $\mathcal T(s)$ ilgili ziyaretlerin zaman kümesi, $T(t)$ o ziyaret sonrasındaki ilk terminaldir. İlk-ziyarette bölüm başına bir geliş, her-ziyarette bütün gelişler alınır. Her ziyaret $i$ için $G_i$ ve $\rho_i=\rho_{t:T(t)-1}$ yazalım:

$$\hat V_O(s)=\frac{\sum_{i=1}^n\rho_iG_i}{n},\qquad
\hat V_W(s)=\frac{\sum_{i=1}^n\rho_iG_i}{\sum_{i=1}^n\rho_i}.$$

Bunlar kaynak (5.5–5.6), **ordinary** ve **weighted/normalized importance sampling**. Ağırlıklı payda 0 ise burada sonuç 0 seçilir. İlk-ziyaret ordinary yansızdır, fakat büyük oranlar örneğin 10 olan tek getiriyi $10G$ yapabilir. Weighted, tek pozitif ağırlıkta $G$ verir; sonlu örneklemde yanlı, uygun koşullarda tutarlıdır. Normalize edilmiş her ağırlık 0…1 aralığındadır. Sınırlı getirilerde kestirim de aynı sınırlar içinde kalır; oranların varyansı sonsuz olsa bile weighted kestiricinin varyansı 0 değerine yakınsar (Precup, Sutton, Dasgupta 2001). İlk-ziyaret ile her-ziyaret ayrımı önemlidir: her-ziyaret biçimlerinin ikisi de sonlu örneklemde yanlı olabilir, uygun koşullarda yanlılık gider. Her-ziyaret ve ordinary biçimleri fonksiyon yaklaşımına daha kolay taşınabildiği için kitapta tutulur.

:::uyarı Kaynak notu: tek örnekte weighted beklentisi
Kaynak s.105, pozitif oranın sadeleştiğini söyleyip beklentiyi $v_b(s)$ diye yorumlar. Bütün davranış yolları pozitif hedef ağırlığı taşıyorsa bu doğru; sıfır oranları varsa koşullamayı ayırmak gerekir. İki eylem de+1 versin, davranış ikisini yarım olasılıkla seçsin, hedef yalnız ilkini seçsin. Tek örnek weighted sonuç, ilk eylemde 1, ikincide payda 0 kuralıyla 0; beklenti 0.5, oysa $v_b=1$. Pozitif ağırlığa koşullanan beklenti de genel olarak koşulsuz $v_b$ değildir. Bu nüans weighted yöntemin sonlu-veri yanlılığını değiştirmez.
:::

**Q için başlangıç oranı farklıdır:** $q_\pi(s,a)$ mevcut eylemi zaten koşullar. $\mathcal T(s,a)$ ziyaretlerinde çarpım **$t+1$ ile başlar**. $\pi(a|s)=0$ olsa bile $q_\pi(s,a)$ tanımlıdır ve ilk eylem sonrası hedef izlenerek tahmin edilebilir. Yanlışlıkla $t$ oranını da katmak weighted ortalamada pozitif sabit olduğunda sadeleşebilir, fakat hedef mevcut eylemi hiç seçmiyorsa bütün veriyi 0 yapar (Alıştırma 5.6).

### İki önem örnekleme deneyi ve sonsuz varyans {#sutton-is-varyans}

Kaynak Şekil 5.3, kullanılabilir aslı oyuncu 13 / krupiye 2 başlangıcını zorlar. Hedef 20/21 toplamında durur; davranış çek/dur yarım olasılıkla seçer.100 tekrar×10000 bölüm, başlangıç kestirimi 0, ortalama kare hata çizilir. Kaynak 100 milyon on-policy getiriyle gerçek değeri yaklaşık **−0.27726** bulur; bağımsız tam-model çözümümüz **−0.277204191825**. Kaynak referansına göre JS deneyinin son MSE değerleri ordinary 0.000956878, weighted 0.000924055 ölçüldü; tek tohumun sıralaması genel üstünlük teoremi değildir. İlk uyumlu getiri±1 olabilir;0 başlangıcı zaten gerçek değere yakınken weighted ilk birkaç örnekle kötüleşebilir (Alıştırma 5.7).

::lab l 5

<figure class="fig wide"><svg viewBox="0 0 640 260" role="img" aria-label="Tek durumlu sonsuz varyans MDP" xmlns="http://www.w3.org/2000/svg"><defs><marker id="Te-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><circle cx="285" cy="100" r="22" fill="var(--paper)" stroke="var(--ink)"/><text x="285" y="105" text-anchor="middle" style="font-size:16px;fill:var(--ink)">s</text><path d="M270,82 C150,4 150,165 270,113" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="143" y="55" text-anchor="middle" style="font-size:16px;fill:var(--ink)">sol: 0.9, R=0</text><path d="M301,116 L420,180" fill="none" stroke="var(--ink)" stroke-width="1.5"/><path d="M310,97 L520,97" fill="none" stroke="var(--ink)" stroke-width="1.5"/><rect x="414" y="175" width="12" height="12" fill="var(--paper)" stroke="var(--ink)"/><rect x="520" y="91" width="12" height="12" fill="var(--paper)" stroke="var(--ink)"/><text x="425" y="153" text-anchor="middle" style="font-size:16px;fill:var(--ink)">sol: 0.1, R=+1</text><text x="466" y="74" text-anchor="middle" style="font-size:16px;fill:var(--ink)">sağ: 1, R=0</text><text x="320" y="225" text-anchor="middle" style="font-size:16px;fill:var(--ink)">π(sol)=1; b(sol)=b(sağ)=0.5; γ=1; vπ=1</text></svg><figcaption>Şekil 4 — Şekil5.4 modelinin yeniden hesaplanan MDP şeması. İki terminale giden ödüller ve döngü açık; E[X]=1, ikinci moment sonsuz (hesaplandı).</figcaption></figure>

Kaynak Örnek 5.5/Şekil 5.4: tek durum, sol eylemde 0.9 olasılıkla 0 ödüllü döngü,0.1 olasılıkla+1 terminal; sağ eylem 0 ödüllü terminal. Hedef hep sol, davranış her eylemi yarım seçer, $\gamma=1$; dolayısıyla hedef değer 1. Sol başarıyla biten $L$ adımlık bir bölümün getirisi 1, oranı $2^L$; diğer bölümlerin oranlı getirisi 0. Kaynak 10 bağımsız ordinary eğriyi 100 milyon bölüme kadar, yatayda log ölçekle gösterir. Deney aynı bütçeyi kullanabilir: ara kontrol noktalarındaki yol-uzunluğu çokterimli sayımları tam dağılımla örneklenir. Bu, bir milyar yolu tek tek yürütmek değildir; çizilen bütün kümülatif noktaların ortak dağılımını koruyan hızlandırmadır. Sonsuz kuyruk kesilip atılmaz. Weighted, ilk pozitif ağırlıktan sonra tam 1 olur; ordinary uzun aralar ve sıçramalar gösterebilir.

:::türetim Sonsuz ikinci momentin açık hesabı
Bir pozitif bölüm için $\Pr(L=\ell,\text{sol başarı})=0.05(0.45)^{\ell-1}$ ve $X=2^\ell$. Böylece

$$\mathbb E[X]=0.1\sum_{j\ge0}0.9^j=1,\qquad
\mathbb E[X^2]=0.2\sum_{j\ge0}1.8^j=\infty.$$

Her-ziyaret düzeltmesi sorunu genel olarak çözmez (Alıştırma 5.8). Böyle bir bölümün bütün ziyaretlerinden gelen ağırlıklı toplamı $2^{L+1}-2$, ziyaret sayısı $L$. Karede üstel $1.8^L$ büyümesi, paydadaki $L^2$ ile bastırılamaz. Sabit sayıda bağımsız bölüm için diğer bölümlerin birer adımlık sıfır katkı olması pozitif olasılıklıdır; aynı alt seri sonsuz ikinci momenti verir.
:::

:::uyarı Kaynak düzeltmesi: sonsuz varyans, yakınsamama demek değildir
Kaynak s.107/Şekil 5.4 ordinary kestirimlerin yakınsamadığını söylüyor. Bu örnekte bağımsız bölümlerin $X_i$ değerleri için $\mathbb E|X_i|=1<\infty$. **Güçlü büyük sayılar yasası** dolayısıyla $n^{-1}\sum_iX_i\to1$ hemen hemen kesin. Sonsuz varyans, standart sonlu-varyans merkezi limit hesabını ve $1/\sqrt n$ hata güvencesini kaldırır; tutarlılığı kaldırmaz.100 milyon bölümdeki düzensiz eğriler yakınsamama kanıtı değildir. Birincil olasılık kaynağı: [Durrett, Probability: Theory and Examples, 5 e, Teorem 2.4.1](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf). Gösterilen on koşunun son değerleri 0.874–1.196 aralığında ölçüldü; sonsuz-varyans iddiası deneyden değil yukarıdaki seriden gelir.
:::

### Ağırlıklı ortalamayı artımlı tutmak {#sutton-is-artimli}

Kaynak (5.7–5.8), $n$ öncesindeki ilk $n-1$ getirinin kestirimini $V_n$ diye adlandırır. Yeni çift $(W_n,G_n)$ eklenince toplam ağırlık $C_n=\sum_{i=1}^nW_i$ ile

$$V_{n+1}=V_n+\frac{W_n}{C_n}(G_n-V_n).$$

Sonraki adımın ağırlık toplamı kaynak indeksinde $C_{n+1}=C_n+W_{n+1}$. Uygulamada daha az indeks şaşırtan biçim `C+=W; V+=(W/C)*(G−V)`; yalnız $C>0$ için. Türetim: yeni pay $C_{old}V_{old}+WG$, yeni payda $C_{old}+W$; eski değeri çıkarınca yukarıdaki fark elde edilir. $W=1$ seçimi $1/n$ ilk-ziyaret ortalamasına döner. Keyfî başlangıç değeri, ilk pozitif ağırlık geldiğinde yeni ortalamadan silinir.

Kaynak s.110 kutusunun **her-ziyaret Q tahmini** karşılığı:

```python
# episode, davranışın seçildiği andaki olasılıkları da saklar.
G, W = 0.0, 1.0
for s, a, r, b_probability in reversed(episode):
    G = r + gamma * G
    C[s, a] += W
    Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
    W *= target_probability(a, s) / b_probability
    if W == 0:
        break
```

İlk güncellemede $W=1$: Q mevcut eylemin oranını içermez. Oran **güncellemeden sonra**, bir önceki zamana geçerken eklenir. $b$ hedefi kapsadığı sürece bölümler arasında değişebilir; sabit hedefin değerlendirmesiyle hedefi de değiştiren kontrolü ayırın. Aşırı çarpımlarda log ağırlıklarıyla $W/C$ oranını kararlı hesaplamak aynı normalizasyonu korur; kırpma ise farklı kestiricidir.

### İndirim ve karar başına oranlar {#sutton-is-ileri}

Kaynak §5.8–5.9, her ödülü son terminale kadar aynı oranla çarpmanın gereksiz varyansını inceler.100 adımlık bölümde $\gamma=0$ için hedef yalnız $R_1$ iken 100 eylem oranını çarpmak sonraki 99 kararı boş yere katar. 0≤γ<1 için indirimi kısmi sonlanma olasılığı olarak yorumlayabiliriz; γ=1 sınırı aşağıdaki sonlu karışımda ayrıca ele alınır.

:::türetim Düz kısmi getiriler ve kaynak (5.9–5.10)
$\bar G_{t:h}=\sum_{k=t+1}^hR_k$ indirimsiz ve terminalden önce kesilebilen getiridir; tanım aralığı $0\le t<h\le T$. $h<T$ için $c_{t,h}=(1-\gamma)\gamma^{h-t-1}$, son $h=T$ için $c_{t,T}=\gamma^{T-t-1}$ tanımlansın. Katsayılar toplamı 1. Her $R_k$ için onu içeren katsayıları toplarsak $\sum_{h=k}^{T}c_{t,h}=\gamma^{k-t-1}$; böylece

$$G_t=\sum_{h=t+1}^{T}c_{t,h}\bar G_{t:h}.$$

Kısmi getiri için yalnız ufuk $h$ sonrasına kadar olan oran gerekir. Ziyaret $i$ için

$$N_i=\sum_{h=t_i+1}^{T_i}c_{t_i,h}\rho_{t_i:h-1}\bar G_{t_i:h},\quad
D_i=\sum_{h=t_i+1}^{T_i}c_{t_i,h}\rho_{t_i:h-1}.$$

İndirim duyarlı ordinary $\sum_iN_i/n$, weighted $\sum_iN_i/\sum_iD_i$ olur. Payda 0 için 0 seçilir. $\gamma=1$ yalnız tam bölüm terimini bırakır; eski önem örneklemesi aynen geri gelir. $\gamma=0$ yalnız ilk ödül ve onun ilk eylem oranını bırakır. İlk-ziyaret ordinary için her terim yansızdır; weighted bir oran kestiricisi olduğundan sonlu-veri yanlılığı olabilir.
:::

:::türetim Karar başına önem örnekleme: kaynak (5.11–5.15)
Tam oranlı getiriyi ödüllere ayıralım:

$$\rho_{t:T-1}G_t=\sum_{k=t+1}^{T}\gamma^{k-t-1}\rho_{t:T-1}R_k.$$

$R_k$ elde edildikten sonraki eylem oranlarını koşullu beklentiyle sondan eleyebiliriz. Geçmiş $H_j$ verildiğinde

$$\mathbb E_b\!\left[\frac{\pi(A_j|S_j)}{b(A_j|S_j)}\middle|H_j\right]
=\sum_a b(a|S_j)\frac{\pi(a|S_j)}{b(a|S_j)}=1.$$

Kapsama altında sıfır-olasılıklı terimler uygun destek üzerinde alınır. Sonlu ufukta veya durdurma/toplam değiştirme için gerekli integrallenebilirlikte kule özelliği,

$$\mathbb E_b[\rho_{t:T-1}R_k]=\mathbb E_b[\rho_{t:k-1}R_k]$$

verir (Alıştırma 5.13). Dolayısıyla

$$\widetilde G_t=\sum_{k=t+1}^{T}\gamma^{k-t-1}\rho_{t:k-1}R_k$$

aynı beklentiye sahiptir. Kaynak (5.15) ordinary kestirimi bu örnekleri ortalar: $V(s)=\sum_{t\in\mathcal T(s)}\widetilde G_t/|\mathcal T(s)|$. Ordinary tahmininde $\rho G$ yerine bu toplamı ortalarız. Varyansı $\gamma=1$ için bile azaltabilir; her MDP için otomatik daha düşük varyans teoremi ileri sürmüyoruz.
:::

:::uyarı Kaynak notu: gelecek oranların bağımsızlığı gerekmez
Kaynak s.114 ifadesini koşulsuz bağımsızlık diye okumayın. İlk ödül sonraki durumu belirlesin; o durumlarda davranış ilk eylemi 0.5 ve 0.25 olasılıkla seçsin, hedef hep ilk eylemi seçsin. Sonraki oran dağılımı sırasıyla 2/0 ve 4/0: önceki ödüle bağlıdır. İki durumda da **koşullu ortalama 1**. Yukarıdaki kanıtın kullandığı özellik budur; olayların sonra olması tek başına bağımsızlık sağlamaz.
:::

Kaynak 2018, bilinen bazı weighted karar-başına önerilerin tutarlı olmadığını bildirir; bu, bütün olası ağırlıklı yöntemlerin imkânsızlık teoremi değildir. **Öğretim uzantısı:** sabit sonlu ufuk $H$ için terminal sonrası sıfır ödülle doldurup her ödül zamanındaki önek oranlarını kendi örneklemleri arasında normalize edin. Her zaman için $\sum_i\rho_{0:k-1}^{(i)}R_k^{(i)}/\sum_i\rho_{0:k-1}^{(i)}$ tutarlı olur; sonlu sayıda indirgenmiş toplam da tutarlıdır. Yalnız hayatta kalan bölümlere normalizasyon yaparak terminal olasılığını silmeyin. Bu ayrı yapı, kaynağın (5.10) karışımıyla aynı formül değildir ve rastgele sonsuz ufka ek koşulsuz genişletilmez.

İndirim duyarlı **Q kontrolü**, mevcut eylemin oranını dışarıda bırakarak [M11.5 x 5](m11-5-sarsa-q.html#x5) içinde uygulanır. Sabit hedefte daha hızlı yineleme: son adımda $D_t=1$, $N_t=R_{t+1}$; önceki adımlarda $w_{t+1}=\pi(A_{t+1}|S_{t+1})/b(A_{t+1}|S_{t+1})$ ile

$$D_t=(1-\gamma)+\gamma w_{t+1}D_{t+1},\qquad
N_t=R_{t+1}D_t+\gamma w_{t+1}N_{t+1}.$$

Hedef geriye tarama içinde değişiyorsa önceden hesaplanan önekleri güncel hedefle karıştırmamak gerekir.

### Kaynak tarihçesi ve değerlendirme sınırları {#sutton-mc-tarih}

“Monte Carlo” adı 1940 yıllarında Los Alamos fizikçilerinin atom bombasıyla ilişkili karmaşık olguları şans oyunları üzerinden çalışmasına uzanır; genel sayısal yöntem kaynakları Kalos–Whitlock (1986), Rubinstein (1981). Bu bölümdeki ilk/her ziyaret ayrımı ve RL bağlantılarının sonuçları Singh–Sutton (1996); Blackjack örneğinin öncülü Widrow–Gupta–Maitra (1973). Dirichlet rastgele-yürüyüş çözümü Kakutani (1945), açıklayıcı kaynaklar Hersh–Griego (1969), Doyle–Snell (1984). Barto–Duff (1994), Curtiss (1954) çözümlemesiyle doğrusal sistemlerin MC çözümünü politika değerlendirmeye bağlayıp büyük problemlerde hesap avantajlarını tartışır.

Politika dışı öğrenme, müdahale ve karşıolgusal çıkarımla da ilişkilidir: Pearl (1995), Balke–Pearl (1994). Weighted önem örneklemesine “normalized” da denir: Koller–Friedman (2009); diğer kaynaklar Rubinstein (1981), Hesterberg (1988), Shelton (2001), Liu (2001). Hedef politika kitabın ilk baskısında “estimation policy” adını taşır. Kaynağın indirim duyarlı sunumu Sutton–Mahmood–Precup–vanHasselt (2014) analizine dayanır; en ayrıntılı geliştirmeyi Mahmood (2017), Mahmood–vanHasselt–Sutton (2014) çalışmalarına bağlar; karar başına yöntem Precup–Sutton–Singh (2000). Sonuncu çalışma politika dışı öğrenmeyi TD, izler ve yaklaşımla da birleştirir; bunların incelikleri sonraki bölümlerin konusudur.

Kaynak 2018, Markov özelliği ihlal edildiğinde MC’nin kimi durumlarda daha az zarar görebileceğini de söyler: hedef başka durumların kestirimlerine dayanmaz. Bu, Markov olmayan bütün süreçlerde garanti değildir; hangi geçmiş/başlangıç dağılımı altında getirilerin ortalandığı önemlidir. İlgi bölgesine hesap ayırma bağlantısı Bölüm 8 içinde sürer. Kaynak 2018 özeti MC ile DP arasındaki iki farkı ayrı tutar: **örnek deneyim** kullanmak ve **önyükleme yapmamak**. Biri diğerini zorunlu kılmaz; TD örnekten öğrenip önyükleme yapar. MC değerlendirmesinde tek duruma odaklanma avantajı, tam bölümleri bekleme maliyetiyle birlikte düşünülür. Kaynakta politika dışı MC için aktarılan araştırma durumu baskının tarihine aittir; bugüne ilişkin taranmış bir literatür iddiası değildir.

### TD, MC ve DP: üç farklı hedefin kaynağı {#sutton-td-uc-yol}

Sutton ve Barto (2018), s.119–140 paketi mevcut MC–TD anlatımını tamamlar; kontrol kısmı [M11.5](m11-5-sarsa-q.html#sutton-td-kontrol) içindedir. TD, MC gibi gerçek deneyimden model olmadan öğrenir; DP gibi henüz öğrenilmekte olan başka değerlere önyükleme yapar. İki özellik birbirine bağlı değildir. Bölüm 7 içindeki n adım, TD ile MC arasında köprü; Bölüm 12 içindeki TD(λ) ortak çerçevedir. Kontrole geçiş her üç yaklaşımda GPI, yani değerlendirme ve politika iyileştirmenin etkileşimidir.

Önceki MC hedefi kaynak (6.1), TD(0) hedefi (6.2) biçiminde geçer. Sabit α MC, değişen ortamlarda yeni getirileri daha çok ağırlıklandırabilir; ancak tam getiriyi öğrenmek için bölüm sonunu bekler. TD(0) **bir sonraki geçişte** günceller. “0”, sıfır adım bakış anlamına gelmez: bu yöntem tek adımlı, λ=0 özel durumudur.

:::türetim Neden iki yöntem de doğru beklentiye yönelir?
Kaynak (6.3–6.4), sabit politika için

$$v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
=\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s]
=\mathbb E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s].$$

MC ilk beklentinin bir örneğini kullanır. DP ikinci beklentiyi tam modelle hesaplar, ama bilinmeyen gerçek ardıl değer yerine güncel V koyar. TD hem beklentiyi örnekler hem gerçek değer yerine V kullanır. Böylece örnekleme ile önyüklemeyi aynı hedefte birleştirir. Bu özdeşlik, her sabit öğrenme oranında bütün rastgele tabloların tam gerçek değerde kalacağı anlamına gelmez.
:::

Kaynak s.120 kutusunun işleyişi, kendi gösterimimizle:

```python
# V nonterminal durumlarda keyfî; bütün terminallerde0.
for episode in episodes_of(policy):
    s = episode.start()
    while not episode.terminal(s):
        a = policy.choose(s)
        next_s, reward, done = episode.step(a)
        target = reward + (0 if done else gamma * V[next_s])
        V[s] += alpha * (target - V[s])
        s = next_s
```

Burada kaynak parametresi 0<α≤1. Robbins–Monro güvencesinde ise α ziyaret sayısıyla değişen bir dizidir; deneyde kullanılan sabit α ile bu teoremi karıştırmayın. Kaynak s.121 backup diyagramı, tek örnek durum–eylem–ardıl durum yoludur. DP bütün ardılların dağılımını açarken TD ve MC **[[örnek güncelleme]]** yapar. MC terminale kadar gider; TD hemen ardıldaki tahmini yaprakta durur.

### TD hatasının zamanı ve değişen tablo {#sutton-td-hata}

Kaynak (6.5) içindeki $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$, t anındaki tahminin hatasıdır; fakat ödül ve ardıl durum ancak **t+1** anında geldiği için o zaman hesaplanabilir. Tablo bölüm boyunca sabitse önceki λ türetiminin tek-adım özdeşliği (6.6) olur:

$$G_t-V(S_t)=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k.$$

Kaynak açıklaması terminalde hem getiri hem V sıfır olduğu için kalan uç terimin yok olduğunu kullanır. Güncel V her adımda değişiyorsa farkı “küçüktür” deyip atmak yerine Alıştırma 6.1 ile açık yazabiliriz.

:::türetim Alıştırma 6.1: çevrimiçi güncellemelerin tam farkı
$V_k$, k adımının güncellemesi **öncesindeki** tablo olsun. Aynı zaman damgalı hata $\delta_k=R_{k+1}+\gamma V_k(S_{k+1})-V_k(S_k)$. Ağırlıklı toplamda komşu terimleri eşleyince, terminal değerleri sabit 0 ise

$$G_t-V_t(S_t)=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
+\sum_{j=t+1}^{T-1}\gamma^{j-t}\left[V_j(S_j)-V_{j-1}(S_j)\right].$$

İkinci toplam eklenmesi gereken farktır. Her terim aynı gelecekteki durumun ardışık iki tablo sürümünü karşılaştırır. Sadece tablo TD(0) kullanılıyorsa bir önceki güncelleme yalnız önceki durumu değiştirir; burada doğrudan katkı ardışık öz döngülerde oluşur. Genel fonksiyon yaklaşımında bir parametre güncellemesi başka durumların değerlerini de değiştirebilir.

Örneğin aynı durum iki kez görülüp terminale gidilsin, iki ödül 1, γ1, V başlangıcı 0, α0.1. İlk hata 1, güncelleme sonrası V0.1; ikinci hata 0.9. TD hata toplamı 1.9, MC hatası 2; ek tablo farkı 0.1 tam eşitliği sağlar. Tablo sabitse ikinci toplam 0 olur.
:::

### Eve dönüş: altı gözlemden iki farklı düzeltme {#sutton-eve-donus}

Kaynak Örnek 6.1 bir **tahmin** problemidir. Cuma 18:00 ofisten çıkarken 30 dakika beklenir. Beş dakika sonra yağmur başlayınca araçtan itibaren 35, toplam 40 dakika tahmin edilir. Otoyol iyi gider;18:20 çıkışında kalan 15, toplam 35 beklenir. Dar yolda kamyon yüzünden gecikilir;18:30 kalan 10,18:40 ev sokağında kalan 3,18:43 evde 0 olur.

| Gözlem | Geçen dakika | Tahmini kalan | Tahmini toplam |
|---|---:|---:|---:|
| Ofisten çıkış | 0 | 30 | 30 |
| Araç, yağmur | 5 | 35 | 40 |
| Otoyoldan çıkış | 20 | 15 | 35 |
| Dar yol, kamyon | 30 | 10 | 40 |
| Evin sokağı | 40 | 3 | 43 |
| Eve varış | 43 | 0 | 43 |

Her geçişin ödülü geçen süredir:5,15,10,10,3; γ1. Değer, beklenen **kalan** süredir. Kaynak dipnotu pozitif süreyi yalnız değerlendirme için kullanır; seyahati kısaltan bir kontrol probleminde ödülü negatif süre seçmek gerekir.

<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px"><svg viewBox="0 0 350 300" role="img" aria-label="MC toplam yolculuk tahminleri ve önerilen değişimler" style="width:100%;max-width:350px;flex:1 1 350px;height:auto"><defs><marker id="c6-drive-MC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><text x="175" y="23" text-anchor="middle" style="font-size:18px;fill:var(--ink)">MC: α=1, γ=1</text><text x="29" y="200" text-anchor="end" style="font-size:13px;fill:var(--ink)">30</text><path d="M35,195 L310,195" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="150" text-anchor="end" style="font-size:13px;fill:var(--ink)">35</text><path d="M35,145 L310,145" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="100" text-anchor="end" style="font-size:13px;fill:var(--ink)">40</text><path d="M35,95 L310,95" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="70" text-anchor="end" style="font-size:13px;fill:var(--ink)">43</text><path d="M35,65 L310,65" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="50" text-anchor="end" style="font-size:13px;fill:var(--ink)">45</text><path d="M35,45 L310,45" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><path d="M35,40 L35,215" fill="none" stroke="var(--ink)" stroke-width="1.5"/><path d="M35,215 L312,215" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="330" y="75" text-anchor="middle" style="font-size:13px;fill:var(--ink)">dk</text><polyline points="40,195 92,95 144,145 196,95 248,65 300,65" fill="none" stroke="var(--ink)" stroke-width="2"/><text x="265" y="58" text-anchor="middle" style="font-size:13px;fill:var(--ink)">gerçek toplam43</text><circle cx="40" cy="195" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M45,195 L45,65" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-MC)"/><text x="40" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">ofisten</text><text x="40" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">çıkış</text><circle cx="92" cy="95" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M97,95 L97,65" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-MC)"/><text x="92" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">araç</text><text x="92" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">yağmur</text><circle cx="144" cy="145" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M149,145 L149,65" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-MC)"/><text x="144" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">otoyol</text><text x="144" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">çıkışı</text><circle cx="196" cy="95" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M201,95 L201,65" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-MC)"/><text x="196" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">dar yol</text><text x="196" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">kamyon</text><circle cx="248" cy="65" r="4" fill="var(--ink)" stroke="var(--ink)"/><text x="248" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">evin</text><text x="248" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">sokağı</text><circle cx="300" cy="65" r="4" fill="var(--ink)" stroke="var(--ink)"/><text x="300" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">eve</text><text x="300" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">varış</text><text x="175" y="281" text-anchor="middle" style="font-size:14px;fill:var(--ink)">durum / gözlem sırası</text></svg><svg viewBox="0 0 350 300" role="img" aria-label="TD toplam yolculuk tahminleri ve önerilen değişimler" style="width:100%;max-width:350px;flex:1 1 350px;height:auto"><defs><marker id="c6-drive-TD" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><text x="175" y="23" text-anchor="middle" style="font-size:18px;fill:var(--ink)">TD: α=1, γ=1</text><text x="29" y="200" text-anchor="end" style="font-size:13px;fill:var(--ink)">30</text><path d="M35,195 L310,195" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="150" text-anchor="end" style="font-size:13px;fill:var(--ink)">35</text><path d="M35,145 L310,145" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="100" text-anchor="end" style="font-size:13px;fill:var(--ink)">40</text><path d="M35,95 L310,95" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="70" text-anchor="end" style="font-size:13px;fill:var(--ink)">43</text><path d="M35,65 L310,65" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><text x="29" y="50" text-anchor="end" style="font-size:13px;fill:var(--ink)">45</text><path d="M35,45 L310,45" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/><path d="M35,40 L35,215" fill="none" stroke="var(--ink)" stroke-width="1.5"/><path d="M35,215 L312,215" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text x="330" y="75" text-anchor="middle" style="font-size:13px;fill:var(--ink)">dk</text><polyline points="40,195 92,95 144,145 196,95 248,65 300,65" fill="none" stroke="var(--ink)" stroke-width="2"/><text x="265" y="58" text-anchor="middle" style="font-size:13px;fill:var(--ink)">gerçek toplam43</text><circle cx="40" cy="195" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M45,195 L45,95" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-TD)"/><text x="40" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">ofisten</text><text x="40" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">çıkış</text><circle cx="92" cy="95" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M97,95 L97,145" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-TD)"/><text x="92" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">araç</text><text x="92" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">yağmur</text><circle cx="144" cy="145" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M149,145 L149,95" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-TD)"/><text x="144" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">otoyol</text><text x="144" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">çıkışı</text><circle cx="196" cy="95" r="4" fill="var(--ink)" stroke="var(--ink)"/><path d="M201,95 L201,65" fill="none" stroke="var(--c-b)" stroke-width="1.5" marker-end="url(#c6-drive-TD)"/><text x="196" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">dar yol</text><text x="196" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">kamyon</text><circle cx="248" cy="65" r="4" fill="var(--ink)" stroke="var(--ink)"/><text x="248" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">evin</text><text x="248" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">sokağı</text><circle cx="300" cy="65" r="4" fill="var(--ink)" stroke="var(--ink)"/><text x="300" y="238" text-anchor="middle" style="font-size:12px;fill:var(--ink)">eve</text><text x="300" y="253" text-anchor="middle" style="font-size:12px;fill:var(--ink)">varış</text><text x="175" y="281" text-anchor="middle" style="font-size:14px;fill:var(--ink)">durum / gözlem sırası</text></svg></div><figcaption>Şekil 5 — Kaynak Şekil6.1,altı durum ve bütün oklar yeniden hesaplandı. MC toplam43 yönüne,TD sonraki tahmine taşır;TD farkları10,−5,5,3,0 (hesaplandı).</figcaption></figure>

Otoyol çıkışında gerçek kalan süre 23, tahmin 15; MC hatası 8 dakika. α0.5 ile değer 4 artıp 19 olur. Kamyonun seyrek bir şanssızlık olması halinde bu bile büyük bir değişim olabilir. MC ancak eve gelince gerçek getiriyi bilir. TD ise sonraki durumun güncel tahminine geçiş süresini ekleyip beklemeden öğrenir; grafikte **toplam** tahminler çizildiği için oklar ardışık toplam tahminlerin farkıdır.

Başka bir günde 25 dakika sonra hâlâ otoyolda kuyrukta olup bir 25 dakika daha bekliyorsanız toplam 50 tahmini oluşur. Eve varmadan eski 30 tahmininin iyimser olduğunu öğrenebilirsiniz. Alıştırma 6.2 bunun daha güçlü bir örneğini ister: yeni bir binaya ve otoparka taşının, ama aynı otoyola bağlanın. Otoyoldan sonraki değerler çok deneyimle zaten iyi biliniyorsa TD yeni başlangıca bu bilgiyi taşır. MC her defasında kalan bütün yolun gürültüsünü yeniden örnekler. Bu avantaj hangi kısmın bilindiğine bağlıdır; “TD her durumda daha iyi” teoremi değildir.

### Çevrimiçi avantaj ve yakınsamanın koşulları {#sutton-td-yakinsama}

TD açık geçiş/ödül modeli gerektirmediği için DP'ye; tek geçişten öğrenebildiği için uzun bölümlerde MC'ye göre avantaj sağlar. Bölümü olmayan sürekli görevlerde de kullanılabilir. Politika dışı MC'de bölümün ilerisine konan keşif eylemleri erken tam getirilerin oranını 0 yapabilir; tek adımlı TD, sonraki bütün kararların uygun olmasını beklemez. Bu, keyfî politika dışı TD'yi veya fonksiyon yaklaşımını otomatik kararlı yapmaz.

Sabit politika için tablo TD(0) güvencesinde yeterli ziyaret, uygun sonlanma veya γ<1, ödül/moment koşulları ve durum başına öğrenme oranlarında toplamın sonsuz, kareler toplamının sonlu olması gerekir. Bazı sonuçlar uygun **doğrusal** yaklaşıma genişler; keyfî doğrusal olmayan ya da politika dışı yaklaşım için aynı güvence yoktur (kaynak Bölüm 9).

:::uyarı Kaynak düzeltmesi: sabit α ile ortalamada tam doğru değer
Kaynak s.124, yeterince küçük sabit α için TD(0) tahminlerinin ortalamada gerçek değere yakınsadığını koşulları ayrıntılandırmadan söyler. Bu ifade genel çevrimiçi Markov veri akışına koşulsuz uygulanamaz. Atıf yapılan [Sutton (1988), Teorem 2](https://jmvidal.cse.sc.edu/library/sutton88a.pdf), ağırlıkların **her tam diziden sonra** güncellendiği özel durumu kanıtlar; dizinin içinde sabit tutulmaları önemlidir.

Çalıştırılan karşı örnek: iki durumlu sabit ödüllü süreçte

$$P=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix},\qquad r=(0,1)^\top,\qquad\gamma=0.9.$$

Gerçek değer $(3.27272727,5.09090909)$. Çevrimiçi sabit α0.1 güncellemesinin durağan **ortalama** tablosu $(3.30319311,5.10246153)$ çıkar. Fark yalnız tek koşunun gürültüsü değildir. Durum ile o zamanki tablo bağımlıdır; bağımsız örnek varsayımı yapılamaz.

Ayrıntı isteyene: $m_j=\mathbb E[V_t\mathbf1_{S_t=j}]$, durağan durum olasılıkları d=(0.6,0.4) olsun. $A_{ij}=I+\alpha e_i(\gamma e_j-e_i)^\top$ ile

$$m'_j=\sum_iP_{ij}\left[A_{ij}m_i+\alpha e_i r_i d_i\right].$$

Dört birinci momentin bu doğrusal sistemini çözmek yukarıdaki ortalamayı verir; bu dört momentin yinelemesindeki4×4 katsayı matrisinin spektral yarıçapı1 altında ölçülür.4096 bağımsız zincir×12000 adım deneyi de ortalamayı yaklaşık $(3.30318634,5.10223722)$ buldu; zincir ortalamalarının standart hataları yaklaşık 0.00134/0.00142. α küçüldükçe bu örnekte sapma küçülür; azalan oran teoremiyle sabit oran deneyini aynı iddia yapmayın.
:::

Kaynak 2018, TD ile MC'nin hangi genel anlamda daha hızlı olduğuna dair evrensel bir matematik sıralama vermediğini söyler. Bu tarihsel araştırma notudur. Aynı paragraf pratikte stokastik görevlerde TD’nin genellikle sabit-α MC’den hızlı öğrenmiş bulunduğunu da aktarır;bu tek yürüyüşten daha geniş bir2018 deneysel değerlendirmesidir,evrensel teorem değildir. Buradaki deneyler seçilen görev ve ölçüt için karşılaştırmadır; hiçbir yöntemin bütün görevlerde daha hızlı olduğu sonucu değildir.

### Beş durumlu yürüyüş: tek koşu ve yedi α eğrisi {#sutton-rastgele-yuruyus}

Kaynak Örnek 6.2 bir [[markov ödül süreci]] (MRP) kullanır: eylem seçimi kaldırılmıştır; sabit politikanın ürettiği zincir de bu biçimde ele alınabilir. BaşlangıçC, her adım eşit olasılıkla sol/sağ; iki uç terminal. Sağdan çıkınca+1, diğer geçişlerde 0 ve γ1. Örnek bölüm C,0,B,0,C,0,D,0,E,1 şeklindedir. Değer sağ uçtan çıkma olasılığıdır: A…E için 1/6,2/6,3/6,4/6,5/6; C değeri 0.5.

<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px"><svg viewBox="0 0 240 240" role="img" aria-label="TD0 bir örnek geçiş backup;R1,gamma0.9,Vsonraki2" style="width:100%;max-width:240px;flex:1 1 240px;height:auto"><defs><marker id="c6-td" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><text x="120" y="25" text-anchor="middle" style="font-size:16px;fill:var(--ink)">TD(0): durumdan duruma</text><circle cx="120" cy="65" r="7" fill="var(--paper)" stroke="var(--ink)"/><circle cx="120" cy="110" r="4" fill="var(--ink)" stroke="var(--ink)"/><circle cx="120" cy="160" r="7" fill="var(--paper)" stroke="var(--ink)"/><path d="M120,72 L120,106" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-td)"/><path d="M120,115 L120,152" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-td)"/><text x="172" y="100" text-anchor="middle" style="font-size:14px;fill:var(--ink)">R=1</text><text x="120" y="192" text-anchor="middle" style="font-size:14px;fill:var(--ink)">V(sonraki)=2</text><text x="120" y="220" text-anchor="middle" style="font-size:14px;fill:var(--ink)">hedef=1+0.9×2=2.8</text></svg><svg viewBox="0 0 490 240" role="img" aria-label="Beş durumlu MRP,C başlangıç,sol0 sağ1 terminal ödülü ve doğru değerler" style="width:100%;max-width:490px;flex:1 1 490px;height:auto"><defs><marker id="c6-rw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><text x="245" y="25" text-anchor="middle" style="font-size:17px;fill:var(--ink)">Eşit olasılıklı sol/sağ yürüyüş; γ=1</text><rect x="23" y="90" width="14" height="14" fill="var(--ink-3)"/><path d="M92,101 L39,101" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><text x="65" y="61" text-anchor="middle" style="font-size:14px;fill:var(--ink)">0</text><circle cx="101" cy="97" r="7" fill="var(--paper)" stroke="var(--ink)"/><text x="101" y="79" text-anchor="middle" style="font-size:15px;fill:var(--ink)">A</text><text x="101" y="152" text-anchor="middle" style="font-size:16px;fill:var(--ink)">1/6</text><path d="M110,93 L163,93" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><path d="M163,101 L110,101" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><text x="136" y="61" text-anchor="middle" style="font-size:14px;fill:var(--ink)">0</text><circle cx="172" cy="97" r="7" fill="var(--paper)" stroke="var(--ink)"/><text x="172" y="79" text-anchor="middle" style="font-size:15px;fill:var(--ink)">B</text><text x="172" y="152" text-anchor="middle" style="font-size:16px;fill:var(--ink)">2/6</text><path d="M181,93 L234,93" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><path d="M234,101 L181,101" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><text x="207" y="61" text-anchor="middle" style="font-size:14px;fill:var(--ink)">0</text><circle cx="243" cy="97" r="7" fill="var(--paper)" stroke="var(--ink)"/><text x="243" y="79" text-anchor="middle" style="font-size:15px;fill:var(--ink)">C</text><text x="243" y="152" text-anchor="middle" style="font-size:16px;fill:var(--ink)">3/6</text><path d="M252,93 L305,93" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><path d="M305,101 L252,101" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><text x="278" y="61" text-anchor="middle" style="font-size:14px;fill:var(--ink)">0</text><circle cx="314" cy="97" r="7" fill="var(--paper)" stroke="var(--ink)"/><text x="314" y="79" text-anchor="middle" style="font-size:15px;fill:var(--ink)">D</text><text x="314" y="152" text-anchor="middle" style="font-size:16px;fill:var(--ink)">4/6</text><path d="M323,93 L376,93" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><path d="M376,101 L323,101" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><text x="349" y="61" text-anchor="middle" style="font-size:14px;fill:var(--ink)">0</text><circle cx="385" cy="97" r="7" fill="var(--paper)" stroke="var(--ink)"/><text x="385" y="79" text-anchor="middle" style="font-size:15px;fill:var(--ink)">E</text><text x="385" y="152" text-anchor="middle" style="font-size:16px;fill:var(--ink)">5/6</text><path d="M394,93 L447,93" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-rw)"/><text x="420" y="61" text-anchor="middle" style="font-size:14px;fill:var(--ink)">1</text><rect x="449" y="90" width="14" height="14" fill="var(--ink-3)"/><text x="243" y="121" text-anchor="middle" style="font-size:15px;fill:var(--ink)">başlangıç C</text><text x="245" y="199" text-anchor="middle" style="font-size:17px;fill:var(--ink)">v(i)=(v(i−1)+v(i+1))/2</text><text x="245" y="226" text-anchor="middle" style="font-size:17px;fill:var(--ink)">Sınırlar0/1; çözüm v(i)=i/6</text></svg></div><figcaption>Şekil 6 — Kaynak s.121 TD backup ve s.125 MRP,örnek geçiş ve Bellman çözümünden üretildi. MRP eylemsiz veya sabit politikanın ürettiği ödüllü zincirdir;iç komşuluklar iki yönlü,terminal uçlara yalnız giriş vardır.</figcaption></figure>

Eski l1 deneyinde “Kaynak Örnek 6.2” görünümü, bütün durumları 0.5 başlatıp MC α0.01/0.02/0.03/0.04 ve TD α0.15/0.1/0.05 eğrilerini 100 bağımsız tekrar×100 bölüm için gösterir. RMS önce beş durum üzerinden, sonra tekrarlar üzerinden ortalanır; ortalama tablonun RMS'i alınmaz. Sol tek koşu paneli 0/1/10/100 bölüm sonrasını gösterir. Sabit α değerleri sonsuza dek son örneklerle dalgalandırır;100 bölümdeki güzel yakınlık tam yakınsama değildir.

Alıştırma 6.3 içindeki kitap çiziminde ilk bölüm yalnız A değerini değiştirir: yol sol terminale gitmiştir. Başlangıçta iç geçişler 0+0.5 hedefiyle değişmez; son geçişte A0.5→0.45 olur. Alıştırma 6.4 için ek geniş α taraması 0.001…1 aralığında 16 nokta kullanır.100 tekrar/100 bölüm ve ilk 100 hatanın ortalaması ölçütünde bu koşunun en iyi MC ayarı α0.03, hata 0.12077; TD α0.1, hata 0.068395. Bu, yalnız son bölümün hatasını en küçükleme ölçütü değildir. Başlatmayı ve ufku değiştirince en iyi ayar da değişebilir.

Alıştırma 6.5: büyük α ile önce düşüp sonra artan hata, başlangıç 0.5 tablosunun gerçek doğrusal profile yakınlığı ve sonraki gürültü tabanı ile ilişkilidir. Gerçek değerlerden başlarsanız hata 0 ile başlar; aşağı inmesi gerekmez. Alıştırma 6.6 için iki kesin yol, tridiagonal Bellman sistemini çözmek veya sınırları 0/1 olan doğrusal harmonik çözümü bulmaktır. Çok sayıda örnekle de yaklaşılabilir; rasyonel değerlerin analitik yolla bulunmuş olması makul tahmindir, yazarların işlem günlüğünü bildiğimiz iddiası değildir.

### Toplu TD'nin kesinlik eşdeğeri {#sutton-batch-sutton}

Kaynak§6.3, elde yalnız 10 bölüm veya 100 geçiş olduğunda aynı veriyi tekrar kullanmayı ele alır. **Toplu güncellemede** her geçişin önerdiği artış aynı donmuş V ile hesaplanır; bütün batch işlenince artışlar toplanıp tablo bir kez değiştirilir. Sonra yeni V ile yeniden geçilir. Bu, bir geçişin güncellemesini hemen sonraki örneğe yansıtmakla aynı sıra değildir. Yeterince küçük α altında aynı veriye tekrar tekrar uygulanan MC ve TD farklı sabit noktalara gider; α yakınsamanın güvenli ölçeğini değiştirir, tekil tanımlı sabit noktayı değiştirmez. Görülmeyen durumlar veya tekil ampirik modeller için ek seçim gerekir; eski l2 bunları 0.5 bırakır.

Kaynak Örnek 6.3/Şekil 6.2, her yeni bölüm sonrasında o ana kadarki **bütün** bölümleri tekrar yakınsatır;100 bağımsız deneyin beş durum RMS ortalaması çizilir. Eski batchmctd/l2, bu limitleri doğrudan çözer. Kaynak bu yürüyüşte batch TD için daha düşük RMS raporlar;buradaki hız sözü örnek verimidir,duvar saati hesabının her zaman daha kısa olduğu iddiası değildir. MC gözlenen durum-getiri çiftlerine kare hatayı en küçükler. TD ise gözlenen geçiş sayılarından kurulan Markov modelinin değerini verir. Bu model doğru kabul edilince başka durumlarda toplanmış bilgi erken duruma taşınabilir.

<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px"><svg viewBox="0 0 360 310" role="img" aria-label="A kesin B,B sekiz gözlemden6kez1ve2kez0terminal;MLEmodeli" style="width:100%;max-width:360px;flex:1 1 360px;height:auto"><defs><marker id="c6-ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><circle cx="45" cy="130" r="7" fill="var(--paper)" stroke="var(--ink)"/><text x="45" y="103" text-anchor="middle" style="font-size:22px;fill:var(--ink)">A</text><circle cx="160" cy="130" r="7" fill="var(--paper)" stroke="var(--ink)"/><text x="160" y="103" text-anchor="middle" style="font-size:22px;fill:var(--ink)">B</text><path d="M54,130 L150,130" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-ab)"/><text x="104" y="112" text-anchor="middle" style="font-size:15px;fill:var(--ink)">R=0;p=1</text><path d="M169,130 L308,45" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-ab)"/><rect x="310" y="38" width="14" height="14" fill="var(--ink-3)"/><text x="260" y="27" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R=1;%75</text><path d="M169,130 L308,215" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-ab)"/><rect x="310" y="208" width="14" height="14" fill="var(--ink-3)"/><text x="260" y="225" text-anchor="middle" style="font-size:16px;fill:var(--ink)">R=0;%25</text><text x="180" y="254" text-anchor="middle" style="font-size:16px;fill:var(--ink)">MC(A)=0; TD(A)=V(B)=6/8=0.75</text><text x="180" y="280" text-anchor="middle" style="font-size:15px;fill:var(--ink)">A gözleminin MC hatası0;</text><text x="180" y="301" text-anchor="middle" style="font-size:15px;fill:var(--ink)">bütün veri hatası0 değildir.</text></svg></div><figcaption>Şekil 7 — Kaynak s.127 sekiz bölümünün en çok olabilirlik modeli. A→B olasılığı1;B son ödül1 olasılığı6/8. Modelde VA=VB=0.75;A için MC0. Bilinen gerçek model verilmiş değildir (hesaplandı).</figcaption></figure>

Kaynak Örnek 6.4, eski el hesabı ve l2 ile aynı sekiz bölümdür: bir A,0,B,0; altı B,1; bir B,0. B sekiz kez görülür, altısı 1 verir; VB=0.75. A için MC yalnız gördüğü 0 getirisini verir, TD kesin A→B gözleminden VA=0.75 verir. MC'nin **A gözlemindeki** hatası 0; bütün veri hatası 0 değildir. Dokuz durum-getiri örneğinde MC kare hata toplamı 1.5, TD için 2.0625 hesaplanır. Kaynağın sıfır hata cümlesi A tahmini bağlamında okunmalıdır. Gerçek süreç verilmediği için gelecekteki gerçek hatayı bu sayılardan ölçtüğümüzü söylemiyoruz.

:::türetim Ampirik model neden TD sabit noktasına eşittir?
$N_i$ i durumundan çıkan geçiş sayısı, $N_{ij}$ i→j sayısı olsun. Toplu TD artışının 0 olması

$$\sum_{\text{i kaynaklı örnekler}}[r+\gamma V(j)-V(i)]=0$$

ve dolayısıyla

$$V(i)=\hat r_i+\gamma\sum_j\frac{N_{ij}}{N_i}V(j)$$

verir. Her i→j kenarının beklenen ödülü,o geçişlerde görülen ödüllerin ortalamasıdır; $\hat r_i=\sum_j(N_{ij}/N_i)\bar r_{ij}$. Bu, örnek frekansları ve ilgili ödül ortalamalarıyla kurulan **en çok olabilirlikli** Markov modelinin Bellman denklemidir. Parametrenin olabilirliği gözlenen veriyi üretme olasılığıdır. Tahmin edilen model kesin doğruymuş gibi değerini hesaplamak “certainty equivalence”, yani kesinlik eşdeğeridir. MC ise aynı durumun gözlenen tam getirilerinin ortalamasını bulur. Sonuçlar özdeş hedefler değildir.
:::

Düz yoğun model n² bellek, geleneksel doğrusal çözüm n³ işlem isteyebilir. TD değer tablosu için n çalışma belleğiyle deneyimi tekrar işleyerek aynı çözümü yaklaşabilir; saklanan veri kümesinin belleği bu n sayısına dahil değildir. Küçük beş durumlu deneyde doğrudan çözüm kolaydır; büyük uzay için avantaj büyür. Çevrimiçi TD/MC tek geçişte bu batch limitlerine tam ulaşmaz, kabaca o yönlere adım atar; bu, evrensel hız üstünlüğü kanıtı değildir.

Politika dışı tek-adım tahmin, Alıştırma 6.7 için $V(S_t)\leftarrow V(S_t)+\alpha\rho_{t:t}\delta_t$ biçimindedir. Koşullu beklentide oran geçişin davranış olasılığını hedefe çevirir, ayrıca $\mathbb E_b[\rho_{t:t}\mid S_t]=1$. Kapsama, ziyaret/öğrenme oranı ve moment koşulları korunur; fonksiyon yaklaşımındaki politika dışı kararsızlığı yalnız bu oran çözmez.

### TD tahmininin tarihçesi ve kapsamı {#sutton-td-tahmin-tarih}

Kaynak 2018, kökleri hayvan öğrenmesi psikolojisi ve yapay zekâya, özellikle Samuel (1959), Klopf (1972) çalışmalarına bağlar. Samuel için§16.2 örnek olay bağlantısı vardır. Holland (1975,1976) değer tahminlerinin tutarlılığı fikri,1970–1975 Michigan Üniversitesi lisansüstü öğrencisi Barto'yu etkiledi; Holland orada öğretim üyesiydi. Booker (1982) ve Holland (1986) bucket brigade bu çizgidedir; kontrol tarihçesi M11.5 içinde tamamlanır.

TD(0), rastgele yürüyüş ve “temporal-difference learning” terimi büyük ölçüde Sutton (1988) kaynaklıdır. DP/MC ilişkisini anlatma biçimi Watkins (1989), Werbos (1987) etkilerini taşır; backup diyagramları kitabın ilk baskısında yeniydi. Sutton'ın ortalamada yakınsama özel sonucu yukarıdaki koşullarıyla; Dayan (1992) olasılık 1 sonucu Watkins–Dayan (1992) temeliyle anılır. Jaakkola–Jordan–Singh (1994), Tsitsiklis (1994) stokastik yaklaşım kuramıyla sonuçları genişletir.

Toplu TD optimalitesi Sutton (1988); Barnard (1993) TD'yi Markov modelini öğrenen bir artımlı adım ile o modelden tahmin hesaplayan bir adımın birleşimi olarak türetir. Kesinlik eşdeğeri terimi uyarlamalı kontrolden gelir: Goodwin–Sin (1984).

TD yalnız ödülle kontrol için değildir: dinamik sistemler hakkında uzun dönem tahmin öğrenme yöntemidir. Kaynak finansal diziler, yaşam süreleri, seçim sonuçları, hava, hayvan davranışı, enerji talebi ve müşteri alışverişini potansiyel uygulama alanları olarak sayar. Bunlar 2018 baskısının araştırma değerlendirmesidir; güncel kullanım yaygınlığı veya alan tavsiyesi iddiası değildir. Yöntemlerin salt tahmin olarak incelenmesi kuramsal özelliklerini anlamayı kolaylaştırmıştır.

### Özet {#ozet}

- RL'de model bilinmez; ajan etkileşerek öğrenir. Ajanlar model, politika ve değer fonksiyonundan hangilerini taşıdıklarına göre sınıflanır: değer tabanlı, politika tabanlı, aktör–eleştirmen; modele dayalı, modelsiz.
- MC değerlendirme tam bölümlerin getirilerini ortalar; ilk ziyaret ya da her ziyaret biçimleri vardır. Getiri ziyaret anından itibaren indirimlenir.
- Artımlı ortalama: yeni = eski + α(hedef − eski). α = 1/n örnek ortalamasıdır, sabit α üstel ağırlıklıdır.
- TD(0): $V(s)\leftarrow V(s)+\alpha[r+\gamma V(s^+)-V(s)]$. Tek adımın deneyimiyle ve bootstrapping ile öğrenir.
- Robbins–Monro: $\sum\alpha=\infty$, $\sum\alpha^2<\infty$; sabit α ile tahmin gürültülü kalır.
- İlk-ziyaret MC ortalaması sabit politika altında uygun koşullarla yansız ve yüksek varyanslıdır; her-ziyaret oranı sonlu veride yanlı olabilir; TD tahmin doğru olana dek yanlı ve düşük varyanslıdır. Rastgele yürüyüşte C'deki hedeflerin varyansları 1/4'e karşı 1/36'dır.
- Toplu güncellemede MC eğitim hatasını en küçükler, TD kesinlik eşdeğeri kestirimini bulur (A–B örneği: V(A) = 0'a karşı 0.75).
- n adımlı getiriler ve λ-getirisi iki uç arasında yanlılık–varyans dengesi kurar. Uygunluk izleri λ-getirisini geri bakışla, adım adım uygular.
