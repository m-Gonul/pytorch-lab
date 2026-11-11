### Değer tabanlıdan politika tabanlıya {#politika-tabanli}

M11.5–M11.6'daki Q-öğrenme ve DQN **modelsiz, değer tabanlı** yöntemlerdir. Politikayı açıkça temsil etmezler; politika $Q$'dan açgözlü olarak çıkar. **Politika gradyanı** yöntemleri ise **modelsiz, politika tabanlıdır**. Politikayı parametreli bir fonksiyon olarak doğrudan öğrenirler ve açık bir değer fonksiyonu temsil etmezler. **Aktör–eleştirmen** yöntemleri ikisini birleştirir: politika (aktör) ve değer fonksiyonu (eleştirmen) birlikte öğrenilir. Bu modül üç aileyi sırayla kurar.

Neden politikayı doğrudan öğrenmeli?

- **Sürekli eylemler.** $\max_aQ(s,a)$ her adımda bir eniyileme problemidir; politika doğrudan bir eylem (ya da dağılım) verir.
- **Rastgele en iyi politikalar.** Durumlar tam ayırt edilemediğinde ya da rakibin olduğu oyunlarda (taş–kağıt–makas) en iyi politika rastgele olabilir. Açgözlü bir değer politikası hep belirlenimcidir.
- **Pürüzsüz değişim.** Politikanın parametresindeki küçük bir değişiklik eylem olasılıklarını da az değiştirir. $Q$'daki küçük bir değişiklik ise açgözlü eylemi birden değiştirebilir.

### Rastgele politika {#rastgele-politika}

Rastgele bir politika $\pi_\theta(a\mid s)=P(a\mid s;\theta)$ durumdan bir eylem seçmez; durum verildiğinde eylemler üzerinde bir **olasılık dağılımı** tanımlar. Bu bir fonksiyondur ve $\theta$ parametreleriyle temsil edilir.

**Sonlu sayıda ayrık eylem** için softmax kullanılır:
$$\pi_\theta(a\mid s)=\frac{\exp(h(s,a;\theta))}{\sum_{a'}\exp(h(s,a';\theta))}.$$
Tercih fonksiyonu $h$, $\theta$'ya göre doğrusal olabilir ($h(s,a;\theta)=\sum_i\theta_if_i(s,a)$, $f_i$ bir taban fonksiyon ailesi) ya da bir sinir ağı olabilir. **Sürekli eylemler** için Gauss politikası kullanılır: $\pi_\theta(a\mid s)=\mathcal N(a\mid\mu(s;\theta),\sigma(s;\theta)^2)$.

Gradyan yöntemleri için gereken nicelik $\nabla_\theta\log\pi_\theta(a\mid s)$'dir. Softmax ve doğrusal $h$ için $\nabla_\theta\log\pi(a\mid s)=f(s,a)-\sum_{a'}\pi(a'\mid s)f(s,a')$ olur: seçilen eylemin özniteliği eksi beklenen öznitelik. Üç eylemli tek-sıcak öznitelikte, $\theta=0$'da eylem 0 seçilmişse bu vektör $(2/3,-1/3,-1/3)$'tür. Gauss için $\partial\log\pi/\partial\mu=(a-\mu)/\sigma^2$ olur; $\sigma=1$, $\mu=1$, $a=1.5$'te $0.5$'tir.

### İki öğrenme kurgusu: taklit ve ödül {#taklit}

**Denetimli öğrenme (taklit).** Bir uzmanın en iyi durum–eylem çiftleri verilmiş olsun: $\{(s_1,a_1^*),\dots,(s_N,a_N^*)\}$. Verinin log-olabilirliği en çoklanır:
$$\theta^*=\arg\max_\theta\sum_i\log\pi_\theta(a_i^*\mid s_i),\qquad\theta_{n+1}\leftarrow\theta_n+\alpha_n\nabla_\theta\log\pi_\theta(a_n^*\mid s_n).$$
Bu, sınıflandırmanın aynısıdır. AlphaGo'nun ilk adımı tam olarak budur (M11.11).

**Pekiştirmeli öğrenme.** Uzman yoktur; elde durum–eylem–ödül üçlüleri vardır. Amaç beklenen indirimli getiriyi en çoklamaktır:
$$J(\theta)=E_{\pi_\theta}\Bigl[\sum_t\gamma^tr_t\Bigr].$$
(Kaynak amacı $\sum_i\gamma^iE_\theta[r_i\mid s_i,a_i]$ biçiminde yazıyor. Durum ve eyleme koşullandığında beklenti artık politikaya bağlı değildir; doğrusu koşulsuz $E_{\pi_\theta}$'dır.) $J$ stokastik gradyan **yükselişiyle** en çoklanır. **REINFORCE** (Williams 1992) güncellemesi:
$$\theta_{n+1}\leftarrow\theta_n+\alpha_n\gamma^nG_n\nabla_\theta\log\pi_\theta(a_n\mid s_n),\qquad G_n=\sum_{t\ge0}\gamma^tr_{n+t}.$$
Taklitteki güncellemeyle aynı biçimdedir. Fark, her eylemin log-olabilirliğinin getirisiyle **ağırlıklandırılmasıdır**. Yüksek getirili eylemler daha olası, düşük getirililer daha az olası hâle gelir. Bu formül politika gradyanı teoreminden çıkar.

### Politika gradyanı teoremi {#teorem}

$J$'nin gradyanı ilk bakışta zordur. Politika değişince hem eylem seçimi hem de **ziyaret edilen durumların dağılımı** değişir, ve bu dağılım bilinmeyen dinamiğe bağlıdır. Teorem bu ikinci etkinin gradyanda hiç görünmediğini söyler:
$$\nabla_\theta J(\theta)\ \propto\ \sum_s\mu_\theta(s)\sum_aQ_\theta(s,a)\,\nabla_\theta\pi_\theta(a\mid s).$$
$\mu_\theta(s)$, politika izlenirken durumların **ziyaret dağılımıdır** (epizodik görevde bölüm başına beklenen ziyaret oranı, sürekli görevde durağan dağılım). $Q_\theta(s,a)$, $s$'de $a$ yapıp sonra $\pi_\theta$'yı izlemenin beklenen indirimli getirisidir. Kaynak sol tarafı $\nabla V_\theta(s)$ diye yazıyor. Sağ tarafta $s$ üzerinden toplam alındığı için sol taraf belirli bir durumun değeri değil, amaç $J(\theta)=V_\theta(s_0)$'dır.

:::türetim Teoremin kanıtı (epizodik durum, γ = 1)
$V(s)=\sum_a\pi(a\mid s)Q(s,a)$ yazılır ve türev alınır:
$$\nabla V(s)=\sum_a\bigl[\nabla\pi(a\mid s)Q(s,a)+\pi(a\mid s)\nabla Q(s,a)\bigr].$$
$Q(s,a)=\sum_{s',r}p(s',r\mid s,a)(r+V(s'))$ ve ödülle geçişler $\theta$'ya bağlı değildir, dolayısıyla $\nabla Q(s,a)=\sum_{s'}p(s'\mid s,a)\nabla V(s')$ olur. Yerine konunca
$$\nabla V(s)=\sum_a\nabla\pi(a\mid s)Q(s,a)+\sum_{s'}P_\pi(s\to s',1)\,\nabla V(s')$$
çıkar. $P_\pi(s\to x,k)$, $\pi$ altında $k$ adımda $s$'den $x$'e gitme olasılığıdır. Aynı açılım $\nabla V(s')$ için tekrar tekrar uygulanır:
$$\nabla V(s_0)=\sum_x\Bigl(\sum_{k=0}^\infty P_\pi(s_0\to x,k)\Bigr)\sum_a\nabla\pi(a\mid x)Q(x,a)=\sum_x\eta(x)\sum_a\nabla\pi(a\mid x)Q(x,a).$$
$\eta(x)$, bir bölümde $x$'e beklenen ziyaret sayısıdır. $\mu(x)=\eta(x)/\sum_{x'}\eta(x')$ ile normalize edilince teorem çıkar; orantı sabiti ortalama bölüm uzunluğudur. Durum dağılımının türevi hiç görünmez, çünkü onun etkisi $Q$'ların içinde zaten hesaba katılmıştır.

Kaynak teoremi "son durum dağılımına yakınsamış bir sistemde $J=\sum_s\mu(s)\sum_aQ(s,a)\pi(a\mid s)$'nin türevi" sezgisiyle açıklıyor. Bu sezgi yanıltıcıdır: o ifadenin türevinde $\nabla\mu$ ve $\nabla Q$ terimleri de vardır. Teoremin asıl içeriği, bu terimlerin toplamının sıfır olmamasına karşın sonucun yalnız $\nabla\pi$ terimiyle yazılabilmesidir.
:::

:::türetim REINFORCE: teoremden Monte Carlo kestiricisine
Toplamlar örneklerle değiştirilir. Durum toplamı, politika izlenirken ziyaret edilen $S_t$ ile örneklenir (indirimli durumda $\gamma^t$ ağırlığıyla). Eylem toplamı için **log-türev hilesi** kullanılır:
$$\sum_a\nabla\pi(a\mid s)Q(s,a)=\sum_a\pi(a\mid s)\,Q(s,a)\,\nabla\log\pi(a\mid s)=E_{a\sim\pi}\bigl[Q(s,a)\nabla\log\pi(a\mid s)\bigr].$$
$Q(S_t,A_t)$'nin yansız bir örneği gerçekleşen getiri $G_t$'dir. Sonuç
$$\nabla J\propto E\bigl[\gamma^tG_t\nabla\log\pi(A_t\mid S_t)\bigr]$$
olur ve bu, REINFORCE güncellemesidir. $\gamma^t$ çarpanı indirimli amacın bir sonucudur: geç adımlardaki durumlar başlangıçtan bakınca daha az önemlidir. Pratikte çoğu uygulama bu çarpanı atar. O zaman kestirici artık tam gradyan değildir, ama genellikle iyi çalışır.
:::

```text
REINFORCE(π_θ):
    θ'yı keyfi başlat
    her bölüm için:
        π_θ ile bir bölüm üret: s_0, a_0, r_0, …, s_T, a_T, r_T
        her adım n = 0, 1, …, T için:
            G_n ← Σ_{t=0}^{T−n} γ^t r_{n+t}
            θ ← θ + α γ^n G_n ∇_θ log π_θ(a_n | s_n)
    döndür π_θ
```

**Kısa koridor** (Sutton ve Barto Örnek 13.1) rastgele politikanın neden gerekli olabileceğini gösterir. Üç durumlu bir koridorda her adım −1 ödül verir; ikinci durumda sağ ve sol eylemleri yer değiştirmiştir. Öznitelikler durumları ayırt etmez, bu yüzden politika her durumda aynı $p=P(\text{sağ})$'yi kullanmak zorundadır. $p=0$ ya da $1$ (belirlenimci) olunca ajan sonsuza dek takılır. ε-açgözlü bir değer politikası iki uçtan birine yakın kalır. En iyi politika $p^*=2-\sqrt2\approx0.59$'dur ve $J=-11.66$ verir (Şekil 1). Politika gradyanı bu iç noktaya doğrudan ulaşır.

<figure class="fig wide"><svg viewBox="0 0 640 256" role="img" aria-label="Kısa koridor örneğinde sağa gitme olasılığına göre beklenen getiri"><path d="M60,24.0 L440,24.0" class="mute"/><text x="54" y="28.0" text-anchor="end" class="sm">0</text><path d="M60,62.0 L440,62.0" class="mute"/><text x="54" y="66.0" text-anchor="end" class="sm">−20</text><path d="M60,100.0 L440,100.0" class="mute"/><text x="54" y="104.0" text-anchor="end" class="sm">−40</text><path d="M60,138.0 L440,138.0" class="mute"/><text x="54" y="142.0" text-anchor="end" class="sm">−60</text><path d="M60,176.0 L440,176.0" class="mute"/><text x="54" y="180.0" text-anchor="end" class="sm">−80</text><path d="M60,214.0 L440,214.0" class="mute"/><text x="54" y="218.0" text-anchor="end" class="sm">−100</text><text x="60" y="230" text-anchor="middle" class="sm">0</text><text x="155.0" y="230" text-anchor="middle" class="sm">0.25</text><text x="250.0" y="230" text-anchor="middle" class="sm">0.5</text><text x="345.0" y="230" text-anchor="middle" class="sm">0.75</text><text x="440" y="230" text-anchor="middle" class="sm">1</text><text x="250.0" y="246" text-anchor="middle" class="sm">P(sağ) = p (her durumda aynı)</text><path d="M67.6,214.0 L71.4,214.0 L75.2,214.0 L79.0,180.0 L82.8,154.7 L86.6,136.7 L90.4,123.1 L94.2,112.6 L98.0,104.2 L101.8,97.4 L105.6,91.7 L109.4,86.8 L113.2,82.7 L117.0,79.1 L120.8,76.0 L124.6,73.3 L128.4,70.9 L132.2,68.7 L136.0,66.8 L139.8,65.0 L143.6,63.4 L147.4,62.0 L151.2,60.7 L155.0,59.5 L158.8,58.4 L162.6,57.4 L166.4,56.4 L170.2,55.6 L174.0,54.8 L177.8,54.0 L181.6,53.3 L185.4,52.7 L189.2,52.1 L193.0,51.6 L196.8,51.0 L200.6,50.6 L204.4,50.1 L208.2,49.7 L212.0,49.3 L215.8,49.0 L219.6,48.6 L223.4,48.3 L227.2,48.1 L231.0,47.8 L234.8,47.6 L238.6,47.3 L242.4,47.1 L246.2,47.0 L250.0,46.8 L253.8,46.7 L257.6,46.5 L261.4,46.4 L265.2,46.3 L269.0,46.3 L272.8,46.2 L276.6,46.2 L280.4,46.2 L284.2,46.1 L288.0,46.2 L291.8,46.2 L295.6,46.3 L299.4,46.3 L303.2,46.4 L307.0,46.5 L310.8,46.7 L314.6,46.9 L318.4,47.1 L322.2,47.3 L326.0,47.5 L329.8,47.8 L333.6,48.1 L337.4,48.5 L341.2,48.9 L345.0,49.3 L348.8,49.8 L352.6,50.4 L356.4,51.0 L360.2,51.7 L364.0,52.5 L367.8,53.4 L371.6,54.4 L375.4,55.5 L379.2,56.8 L383.0,58.3 L386.8,60.0 L390.6,62.0 L394.4,64.3 L398.2,67.1 L402.0,70.4 L405.8,74.6 L409.6,79.8 L413.4,86.5 L417.2,95.4 L421.0,108.0 L424.8,126.9 L428.6,158.5 L432.4,214.0" class="ca w2"/><circle cx="282.6" cy="46.1" r="4.5" class="cb-f"/><text x="290.59884629822386" y="38.148023074035535" class="sm">p* = 2 − √2 ≈ 0.586, J = −11.66</text><circle cx="421.0" cy="108.0" r="3.5" class="ca-f"/><circle cx="79.0" cy="180.0" r="3.5" class="ca-f"/><text x="60" y="14" class="sm">Kısa koridor (eylemleri ters ikinci durumlu): başlangıçtan beklenen getiri J(p)</text><text x="470" y="36" class="sm">Durumlar ayırt edilemiyor:</text><text x="470" y="52" class="sm">politika her durumda aynı</text><text x="470" y="68" class="sm">p'yi kullanmak zorunda.</text><text x="470" y="84" class="sm"></text><text x="470" y="100" class="sm">p = 0 ya da 1 (belirlenimci):</text><text x="470" y="116" class="sm">  ajan sonsuza dek takılır, J → −∞.</text><text x="470" y="132" class="sm">ε = 0.1 ile ε-açgözlü (mavi):</text><text x="470" y="148" class="sm">  J = −44.2 ya da −82.1.</text><text x="470" y="164" class="sm">En iyi rastgele politika: −11.66.</text><text x="470" y="180" class="sm"></text><text x="470" y="196" class="sm">Q tabanlı açgözlü politika bu</text><text x="470" y="212" class="sm">en iyiye ulaşamaz; politika</text><text x="470" y="228" class="sm">gradyanı ulaşır.</text></svg><figcaption>Şekil 1 — Rastgele politikanın en iyi olabildiği bir örnek (Sutton ve Barto 13.1). J(p) = (2p − 4)/(p(1 − p)), doğrusal sistemle de çözüldü; en iyi nokta p* = 2 − √2'de −11.66'dır.</figcaption></figure>

::lab l1

### Taban çizgisi ve avantaj {#taban}

REINFORCE yansızdır ama varyansı yüksektir. Bütün getiriler pozitif büyük bir sayıysa, her güncelleme seçilen eylemin olasılığını artırır; yalnız artış miktarları farklıdır. Doğru yönü bulmak için çok örnek gerekir.

**Taban çizgisi** $b(s)$, getiriden çıkarılır. Eylemden bağımsız olduğu sürece gradyanı değiştirmez:
$$\nabla J\propto\sum_s\mu(s)\sum_a\bigl(Q(s,a)-b(s)\bigr)\nabla\pi(a\mid s),$$
çünkü
$$\sum_ab(s)\nabla\pi(a\mid s)=b(s)\nabla\sum_a\pi(a\mid s)=b(s)\nabla1=0.$$
Beklenti aynı kalır, ama **varyans** büyük ölçüde değişebilir. Aşağıdaki deney bunu bir haydut üzerinde kapalı biçimde hesaplıyor. Ödüller 5 civarında toplanmışken $b=0$ ile varyans, en iyi taban çizgisine göre yaklaşık 30 kat büyüktür. Varyansı en küçükleyen sabit taban çizgisi $b^*=E[r\|\nabla\log\pi\|^2]/E[\|\nabla\log\pi\|^2]$'dir ve ortalama ödüle çok yakındır.

::lab l2

Yaygın seçim $b(s)\approx V^\pi(s)$'dir. **Avantaj fonksiyonu**
$$A(s,a)=Q(s,a)-V^\pi(s)$$
şu soruya cevap verir: değeri $V^\pi(s)$ olan $s$ durumunda $a$'yı seçersem ortalamadan ne kadar fazla kazanırım? Güncelleme $\theta\leftarrow\theta+\alpha\gamma^nA(s_n,a_n)\nabla\log\pi(a_n\mid s_n)$ olur. Ortalamadan iyi eylemler güçlenir, kötüler zayıflar. Deneysel yakınsama çok daha hızlıdır.

**Taban çizgili REINFORCE** iki ağ öğrenir: politika ağı $\pi_\theta$ ve değer ağı $V_w$. $\delta=G_n-V_w(s_n)$, avantajın tek örnekli bir kestirimidir:

```text
REINFORCE-taban(π_θ, V_w):
    her bölüm için:
        π_θ ile bir bölüm üret
        her adım n için:
            G_n ← Σ_{t=0}^{T−n} γ^t r_{n+t}
            δ ← G_n − V_w(s_n)
            w ← w + α_w δ ∇_w V_w(s_n)
            θ ← θ + α_θ γ^n δ ∇_θ log π_θ(a_n | s_n)
```

Kaynaktaki sözde kod değer ağının güncellemesine de $\gamma^n$ çarpanı koyuyor. Değer ağının işi her ziyaret edilen durumda $G_n$'ye regresyondur. $\gamma^n$ çarpanı geç durumların regresyonunu keyfî olarak zayıflatır; Sutton ve Barto'nun algoritmasında eleştirmen güncellemesinde bu çarpan yoktur. Aynı not aşağıdaki aktör–eleştirmen sözde koduna da uygulanır.

### Aktör–eleştirmen {#aktor-elestirmen}

Taban çizgili REINFORCE yine bölüm sonunu bekler ve $G_n$'nin varyansını taşır. Değer ağını Monte Carlo yerine **TD** ile güncellemek ve avantajı TD hatasıyla kestirmek bir sonraki adımdır:
$$\delta=r_n+\gamma V_w(s_{n+1})-V_w(s_n).$$
$V_w=V^\pi$ olduğunda $E[\delta\mid s_n,a_n]=Q(s_n,a_n)-V(s_n)=A(s_n,a_n)$ olur: TD hatası avantajın yansız bir tek örnekli kestirimidir. $V_w$ yaklaşık olduğu sürece yanlıdır, ama varyansı $G_n-V(s_n)$'ninkinden çok düşüktür ve güncelleme her adımda yapılabilir. Değer ağı **eleştirmen**, politika **aktördür**.

```text
Aktör-Eleştirmen(π_θ, V_w):
    her bölüm için:
        s ← s_0;  I ← 1
        s son durum olmadığı sürece:
            a ~ π_θ(· | s);  uygula, r ve s' gözle
            δ ← r + γ V_w(s') − V_w(s)        (s' son durumsa V_w(s') = 0)
            w ← w + α_w δ ∇_w V_w(s)
            θ ← θ + α_θ I δ ∇_θ log π_θ(a | s)
            I ← γ I;  s ← s'
```

Aktör–eleştirmen GPI'nin (M11.3) modelsiz bir biçimidir. Eleştirmen mevcut politikayı değerlendirir, aktör eleştirmenin söylediği yönde politikayı biraz iyileştirir. Bu ikisi aynı anda, adım adım yapılır.

### A2C {#a2c}

**Avantaj aktör–eleştirmeni** (A2C; asenkron sürümü A3C, Mnih ve ark. 2016) bu fikrin derin öğrenme sürümüdür. Literatürdeki standart biçimi şudur:

- Eleştirmen bir **durum değeri** $V_w(s)$'dir.
- Avantaj $n$ adımlı getiriyle kestirilir: $\hat A_t=\sum_{k=0}^{n-1}\gamma^kr_{t+k}+\gamma^nV_w(s_{t+n})-V_w(s_t)$.
- Birçok paralel ortamdan gelen deneyimle eşzamanlı güncelleme yapılır.
- Politikanın erken çökmesini önlemek için bir **entropi bonusu** eklenir: $\beta\,\mathcal H(\pi_\theta(\cdot\mid s))$.

:::uyarı Kaynak düzeltmesi (lect_9): A2C ve DPG sözde kodlarında max
Kaynaktaki A2C bir Q-eleştirmeni kullanıyor ve hem TD hedefini hem avantajı $r+\gamma\max_{a'}Q_w(s',a')$ ile kuruyor. Aktör–eleştirmenin eleştirmeni **mevcut politikanın** değerini öğrenmelidir. $\max$, $Q^\pi$ yerine en iyi politikanın $Q^*$'ını hedefler, yani bir Q-öğrenme hedefidir. Politika gradyanı teoremi $Q^\pi$ ister; $\max$'lı hedefle kestirilen avantaj yanlıdır. Politika üstü doğru hedef $r+\gamma\sum_{a'}\pi_\theta(a'\mid s')Q_w(s',a')$ (beklenen SARSA) ya da örneklenen $a'$ ile $r+\gamma Q_w(s',a')$'dır. Mnih ve ark.'nın A2C/A3C'si zaten Q değil $V$ eleştirmeni ve $n$ adımlı avantaj kullanır.

Aynı sorun DPG sözde kodunda daha belirgindir. Eylemler süreklidir ve $\max_{a'}Q_w(s',a')$ her adımda çözülemeyen bir eniyileme problemidir. DPG'nin hedefi deterministik politikanın kendi eylemidir: $r+\gamma Q_w(s',\mu_\theta(s'))$. Ayrıca deterministik politika kendiliğinden keşif yapmaz. Silver ve ark. (2014) ve DDPG (Lillicrap ve ark. 2016), davranışa gürültü ekleyip eleştirmeni politika dışı öğrenir.
:::

### Deterministik politika gradyanı {#dpg}

Sürekli eylem uzayında deterministik bir politika $\mu_\theta(s)\to a$ için gradyan teoremi daha da basittir (Silver ve ark. 2014):
$$\nabla_\theta J\ \propto\ \sum_s\mu^\pi(s)\,\nabla_\theta\mu_\theta(s)\,\nabla_aQ(s,a)\big|_{a=\mu_\theta(s)}=E_{s\sim\mu^\pi}\Bigl[\nabla_\theta\mu_\theta(s)\,\nabla_aQ(s,a)\big|_{a=\mu_\theta(s)}\Bigr].$$
Zincir kuralıdır: politika, eylemi eleştirmenin $Q$'yu artırdığı yöne doğru iter. Eylemler üzerinde integral gerekmez. Bu yüzden yüksek boyutlu eylem uzaylarında rastgele politika gradyanından daha az örnek ister. Örnek: $Q(s,a)=-(a-2)^2$, $\mu_\theta(s)=\theta s$, $s=1$, $\theta=0.5$ için $\nabla_\theta\mu=s=1$, $\partial_aQ=-2(0.5-2)=3$, gradyan $3$'tür.

```text
DPG(μ_θ, Q_w):
    her bölüm için:  s ← s_0
        s son durum olmadığı sürece:
            a ← μ_θ(s) + keşif gürültüsü;  uygula, r ve s' gözle
            δ ← r + γ Q_w(s', μ_θ(s')) − Q_w(s, a)
            w ← w + α_w δ ∇_w Q_w(s, a)
            θ ← θ + α_θ ∇_θ μ_θ(s) ∇_a Q_w(s, a)|_{a = μ_θ(s)}
            s ← s'
```

Aşağıdaki deney DPG'yi tek boyutlu bir doğrusal–karesel düzenleyicide (LQR) çalıştırıyor. Sistem $s'=s+a$, ödül $-(s^2+\rho a^2)$, politika $a=-ks$'dir. Bu problemin kesin çözümü Riccati denkleminden bilinir. DPG gradyanı ($Q$'nun kesin türeviyle) Riccati kazancına yakınsar. Büyük bir adım ise kazancı kararlılık bölgesinin dışına atabilir.

::lab l3

**DDPG** (Lillicrap ve ark. 2016) DPG'yi DQN'in hileleriyle (deneyim tekrarı, yavaş güncellenen hedef ağlar) birleştirir ve sürekli kontrol problemlerinde (robot kolu, yürüme) kullanılır. TD3 ve SAC onun kararlılık sorunlarını giderir.

**CartPole.** Gymnasium'daki CartPole ortamında bir direk, sürtünmesiz bir ray üzerinde hareket eden bir arabaya eyleyicisiz bir mafsalla bağlıdır. Direk dik başlar. Amaç, arabayı sola ya da sağa iterek direği dengede tutmaktır. Eylemler: 0 sola it, 1 sağa it. Gözlem dört sayıdır: arabanın konumu, hızı, direğin açısı ve açısal hızı. Her adım +1 ödül verir; direk 12°'yi aşınca ya da araba raydan çıkınca bölüm biter. Bu modülün e aşaması CartPole'u paralel ortamlı A2C ile çözüyor. Taban çizgisiz REINFORCE'un öğrenme eğrisi taban çizgilisininkinden çok daha gürültülüdür.

### Özet {#ozet}

- Değer tabanlı (Q-öğrenme), politika tabanlı (politika gradyanı) ve ikisini birleştiren aktör–eleştirmen yöntemleri vardır. Politika tabanlı yöntemler sürekli eylemleri ve rastgele en iyi politikaları doğrudan ele alır.
- Rastgele politika: ayrık eylemde softmax ($\nabla\log\pi$ = öznitelik − beklenen öznitelik), sürekli eylemde Gauss ($\partial\log\pi/\partial\mu=(a-\mu)/\sigma^2$).
- Taklit log-olabilirliği en çoklar. RL'de amaç $J(\theta)=E_\pi[\sum\gamma^tr_t]$'dir ve REINFORCE onu getiri ağırlıklı log-olabilirlik gradyanıyla en çoklar.
- Politika gradyanı teoremi: $\nabla J\propto\sum_s\mu(s)\sum_aQ(s,a)\nabla\pi(a\mid s)$; durum dağılımının türevi gerekmez. Log-türev hilesiyle Monte Carlo kestiricisi REINFORCE'tur.
- Kısa koridorda en iyi politika rastgeledir: $p^*=2-\sqrt2$, $J=-11.66$.
- Taban çizgisi yanlılık getirmez ($\sum_ab\nabla\pi=0$) ve varyansı çok azaltır; $b\approx V$ ile avantaj $A=Q-V$ çıkar. Değer ağı $\gamma^n$ çarpanı olmadan regresyonla öğrenilir.
- Aktör–eleştirmen: $\delta=r+\gamma V_w(s')-V_w(s)$ avantajın düşük varyanslı (yanlı) bir kestirimidir. Standart A2C, $V$ eleştirmeni, $n$ adımlı avantaj, paralel ortamlar ve entropi bonusu kullanır; eleştirmen hedefinde $\max$ kullanılmaz.
- DPG: $\nabla J\propto E[\nabla_\theta\mu_\theta(s)\nabla_aQ|_{a=\mu(s)}]$. Hedefte $Q(s',\mu(s'))$ kullanılır, keşif gürültüyle sağlanır. LQR'de Riccati kazancına yakınsar.
