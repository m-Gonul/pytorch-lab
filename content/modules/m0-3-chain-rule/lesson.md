### Neden türev? Problem tanımı {#neden}

Bir sinir ağını eğitmek, milyonlarca parametresi olan bir kayıp fonksiyonu $L(\theta)$'yı küçültmektir. Her adımda şu soruyu yanıtlamamız gerekir: **her parametreyi biraz değiştirirsem kayıp ne kadar değişir?** Bu sorunun cevabı türevdir; tüm parametreler için birlikte gradyan $\nabla_\theta L$. Gradyan inişi (M1.1) parametreyi gradyanın tersine küçük bir adım götürür: $\theta\leftarrow\theta-\eta\nabla_\theta L$.

İki zorluk var: (1) $L$, yüzlerce basit işlemin iç içe geçmiş bileşkesidir, kapalı formda türevini yazmak pratik değildir; (2) parametre sayısı çok büyüktür, her biri için ayrı hesap yapmak çok yavaştır. Bu modülün cevabı: **zincir kuralı** bileşkenin türevini parçaların yerel türevlerinden kurar, **ters mod (geri yayılım)** de bütün parametrelerin türevini tek bir geri geçişte, kabaca ileri geçişin birkaç katı maliyetle verir.

Yol haritası: tek değişkenli türev → doğrusal yaklaşım → kısmi türev ve gradyan → Jacobian → zincir kuralı → hesap grafiği ve geri yayılım → ileri mod ve ters mod → sayısal türev ve gradyan kontrolü → PyTorch'ta otomatik türev.

### Tek değişkenli türev: sekanttan teğete {#turev}

:::sezgi
Arabanın 2 saatte 150 km gittiğini biliyorsan ortalama hız 75 km/sa'tir. Hız göstergesinin o anki değeri ise "çok kısa bir zaman aralığında ortalama hız"dır. Türev, aralığı sıfıra küçülttüğümüzde ortalama değişim oranının yaklaştığı değerdir.
:::

:::tanım Türev
$f:\mathbb R\to\mathbb R$, $x$ noktasında **türevlenebilir**dir ancak ve ancak şu limit vardır (sonlu):
$$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.$$
Pay $\Delta f$ (çıktının değişimi), payda $\Delta x=h$ (girdinin değişimi): **birim** girdi değişimi başına çıktı değişimi. Birimi $[f]/[x]$ (km/sa, kayıp/parametre). Geometride: $(x,f(x))$ ile $(x+h,f(x+h))$'den geçen **sekantın** eğimi, $h\to0$ iken **teğetin** eğimine gider.
:::

Limitin var olması için $h$'nin hem sağdan hem soldan sıfıra gitmesi aynı sonucu vermelidir. $|x|$'in $0$'da sağdan eğimi $+1$, soldan $-1$'dir: türev yoktur. ReLU $=\max(0,x)$ da $0$'da türevsizdir (sağ 1, sol 0). Bu noktalarda ne yapıldığını aşağıda konuşacağız.

<figure class="fig"><svg viewBox="0 0 420 250" role="img" aria-label="Bir eğri, x noktasındaki teğet ve h küçüldükçe teğete yaklaşan sekantlar"><defs><marker id="ma3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><line x1="30" y1="225" x2="405" y2="225" class="ink" marker-end="url(#ma3)"/><line x1="40" y1="238" x2="40" y2="12" class="ink" marker-end="url(#ma3)"/><polyline points="40.0,203.0 52.0,202.8 64.0,202.1 76.0,201.1 88.0,199.5 100.0,197.6 112.0,195.2 124.0,192.4 136.0,189.2 148.0,185.5 160.0,181.4 172.0,176.9 184.0,171.9 196.0,166.5 208.0,160.7 220.0,154.4 232.0,147.7 244.0,140.6 256.0,133.0 268.0,125.0 280.0,116.6 292.0,107.7 304.0,98.5 316.0,88.7 328.0,78.6 340.0,68.0 352.0,57.0 364.0,45.5 376.0,33.7 388.0,21.3 400.0,8.6" class="ink w2" fill="none"/><line x1="56.0" y1="250.0" x2="408.0" y2="17.7" class="cb dash"/><line x1="56.0" y1="234.4" x2="408.0" y2="54.9" class="cb dash"/><line x1="56.0" y1="218.8" x2="408.0" y2="92.1" class="cp w2"/><circle cx="160.0" cy="181.4" r="4.5" class="ink-f"/><circle cx="360.0" cy="49.4" r="4" class="cb-f"/><circle cx="260.0" cy="130.4" r="4" class="cb-f"/><line x1="160.0" y1="225" x2="160.0" y2="181.4" class="mute dash"/><text x="156.0" y="240" class="sm">x</text><text x="312.0" y="47.4" class="sm tb">x+h₁</text><text x="266.0" y="146.4" class="sm tb">x+h₂</text><text x="300" y="145.8" class="sm tp">teğet: eğim f′(x)</text><text x="70" y="40" class="sm tb">sekantlar: eğim (f(x+h) − f(x))/h</text></svg><figcaption>Şekil 1 — h küçüldükçe (h₁ → h₂ → 0) sekant (turuncu kesikli) teğete (yeşil) yaklaşır. Türev, bu limitin eğimidir.</figcaption></figure>

**Yaygın türevler ve kurallar.** Bunları türetebilmek, ezberlemekten daha önemlidir; her biri limit tanımından ya da aşağıdaki kurallardan gelir.

| fonksiyon | türev | not |
|---|---|---|
| $x^n$ | $nx^{n-1}$ | binom açılımından |
| $e^x$ | $e^x$ | kendi türevine eşit tek (ölçek dışında) fonksiyon |
| $\ln x$ | $1/x$ | $x>0$ |
| $\sin x$, $\cos x$ | $\cos x$, $-\sin x$ | radyan cinsinden |
| $\sigma(x)=\frac1{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | en fazla 1/4 (x = 0'da) |
| $\tanh x$ | $1-\tanh^2x$ | en fazla 1 |
| $\mathrm{ReLU}(x)$ | $1$ ($x>0$), $0$ ($x<0$) | $x=0$'da tanımsız |
| $\mathrm{softplus}(x)=\ln(1+e^x)$ | $\sigma(x)$ | ReLU'nun yumuşak hâli |

| kural | formül |
|---|---|
| doğrusallık | $(af+bg)'=af'+bg'$ |
| çarpım | $(fg)'=f'g+fg'$ |
| bölüm | $(f/g)'=(f'g-fg')/g^2$ |
| zincir | $(f\circ g)'(x)=f'(g(x))\,g'(x)$ |

:::türetim Sigmoid'in türevi
$\sigma(x)=(1+e^{-x})^{-1}$. Zincir kuralıyla: $\sigma'(x)=-(1+e^{-x})^{-2}\cdot(-e^{-x})=\frac{e^{-x}}{(1+e^{-x})^2}$. Şimdi $\frac{e^{-x}}{1+e^{-x}}=1-\frac1{1+e^{-x}}=1-\sigma(x)$ olduğunu kullanalım: $\sigma'(x)=\frac1{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}=\sigma(x)(1-\sigma(x))$. **Pratik sonuç:** geri geçişte sigmoid'in türevi için yeniden üs hesaplamaya gerek yok; ileri geçişte bulunan $\sigma(x)$ yeter. En büyük değer $x=0$'da $1/4$; $|x|$ büyüdükçe sıfıra yaklaşır — derin sigmoid ağlarında gradyanın sönmesinin nedeni (M3.5).
:::

### Türev = en iyi doğrusal yaklaşım {#dogrusal-yaklasim}

Türevi bir eğim olarak değil, bir **yaklaşım** olarak düşünmek, çok boyuta genelleştirmenin anahtarıdır:
$$f(x+h)=f(x)+f'(x)\,h+r(h),\qquad \frac{r(h)}{h}\xrightarrow{h\to0}0.$$
Yani $x$'in yakınında $f$, eğimi $f'(x)$ olan doğruya çok benzer; hata $r(h)$, $h$'den **daha hızlı** sıfıra gider. Bu, türev tanımının yeniden yazılmış hâlidir (her iki yanı $h$'ye böl, limit al).

$f$ iki kez türevlenebilirse **Taylor açılımı** hatanın büyüklüğünü de verir:
$$f(x+h)=f(x)+f'(x)h+\tfrac12f''(x)h^2+\tfrac16f'''(x)h^3+\dots$$
İkinci terim **eğrilik**tir: $f''>0$ ise fonksiyon yukarı kıvrılır (çanak), doğrusal yaklaşım değeri olduğundan küçük tahmin eder. Gradyan inişinin neden bazen fazla, bazen az adım attığını (M1.2) bu terim açıklar.

:::örnek Sayılarla
$f(x)=x^2$, $x=3$, $h=0.1$: gerçek $f(3.1)=9.61$. Doğrusal yaklaşım: $9+6\cdot0.1=9.6$. Hata $0.01=\tfrac12\cdot2\cdot0.1^2$ — tam olarak ikinci dereceden terim ($x^2$'nin üçüncü türevi yok).
:::

Aşağıdaki deneyde sekantın teğete nasıl yaklaştığını ve **sayısal türevin** neden çok küçük $h$'de bozulduğunu göreceksin (ayrıntısı "Sayısal türev" bölümünde).

::lab l1

### Kısmi türev ve gradyan {#gradyan}

Çok değişkenli $f:\mathbb R^n\to\mathbb R$ (kayıp fonksiyonu gibi) için her değişkene ayrı ayrı bakabiliriz: diğer değişkenleri **sabit tutup** yalnız $x_i$'yi değiştirmek.

:::tanım Kısmi türev, gradyan
$$\frac{\partial f}{\partial x_i}(x)=\lim_{h\to0}\frac{f(x+he_i)-f(x)}{h},\qquad \nabla f(x)=\Big(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\Big).$$
$e_i$, $i$. koordinat yönündeki birim vektör. Gradyan, girdiyle **aynı shape**'te bir vektördür: parametre tensörü (64, 128) ise gradyanı da (64, 128). PyTorch bu yüzden `w.grad`'ı `w` ile aynı shape'te tutar.
:::

:::örnek
$f(x,y)=x^2y+3y$. $y$'yi sabit tutarak $\partial f/\partial x=2xy$; $x$'i sabit tutarak $\partial f/\partial y=x^2+3$. $(1,2)$'de $\nabla f=(4,4)$.
:::

**Doğrusal yaklaşım çok boyutta:** $f(x+\delta)\approx f(x)+\nabla f(x)\cdot\delta$. Değişim, gradyan ile adımın **iç çarpımıdır** (M0.1). Buradan iki önemli sonuç çıkar.

:::türetim Yönlü türev ve en dik çıkış yönü
$u$ birim vektör ($\|u\|=1$) olsun. $u$ yönünde birim adım başına değişim, **yönlü türev**: $D_uf(x)=\lim_{t\to0}\frac{f(x+tu)-f(x)}{t}$. Doğrusal yaklaşımdan $f(x+tu)-f(x)\approx t\,\nabla f\cdot u$, dolayısıyla
$$D_uf=\nabla f\cdot u=\|\nabla f\|\,\|u\|\cos\varphi=\|\nabla f\|\cos\varphi.$$
1. **En büyük** değer $\varphi=0$'da: $u=\nabla f/\|\nabla f\|$. Gradyan, fonksiyonun **en hızlı arttığı** yöndür ve o yöndeki artış hızı $\|\nabla f\|$'dir. (Bu, M0.1'deki Cauchy–Schwarz eşitsizliğinin eşitlik durumudur: $|\nabla f\cdot u|\le\|\nabla f\|$.)
2. **En küçük** değer $\varphi=180°$'de: $-\nabla f$ en dik iniş yönüdür — gradyan inişinin yönü.
3. $\varphi=90°$'de $D_uf=0$: gradyana dik yönlerde $f$ (birinci derecede) değişmez. Bu yönler **eş-yükselti eğrisine** (kontur) teğettir; yani **gradyan konturlara diktir**.

**Koşullar:** $f$ $x$'te türevlenebilir olmalı (kısmi türevlerin varlığı tek başına yetmez; kısmi türevler sürekliyse yeter). "En dik iniş" yalnız **yerel** ve Öklid uzunluğuna göredir: başka bir uzaklık ölçüsünde (ör. parametreler farklı ölçekteyse) en dik yön değişir — Adam gibi yöntemlerin ön-koşullamasının (M1.2) sebebi.
:::

<figure class="fig"><svg viewBox="0 0 420 250" role="img" aria-label="Eş-yükselti elipsleri, bir noktada konturlara dik gradyan oku ve teğet yön"><defs><marker id="mb3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker><marker id="mc3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker></defs><g class="mute" fill="none"><ellipse cx="200" cy="125" rx="40" ry="20"/><ellipse cx="200" cy="125" rx="90" ry="45"/><ellipse cx="200" cy="125" rx="140" ry="70"/><ellipse cx="200" cy="125" rx="190" ry="95"/></g><circle cx="200" cy="125" r="3" class="ink-f"/><text x="206" y="122" class="sm">minimum</text><circle cx="263.6" cy="156.8" r="4" class="ink-f"/><line x1="263.6" y1="156.8" x2="295" y2="220" class="cb w2" marker-end="url(#mb3)"/><line x1="263.6" y1="156.8" x2="232" y2="93" class="ca w2" marker-end="url(#mc3)"/><line x1="230" y1="173" x2="300" y2="139" class="cp dash w2"/><text x="300" y="212" class="sm tb">∇f (en dik çıkış)</text><text x="150" y="95" class="sm ta">−∇f (en dik iniş)</text><text x="302" y="138" class="sm tp">kontura teğet: D_u f = 0</text></svg><figcaption>Şekil 2 — Kontur çizgileri (gri elipsler) üzerinde f sabittir. Gradyan (turuncu) konturlara diktir ve yukarı bakar; −∇f (mavi) en dik iniş yönüdür ama minimumu doğrudan göstermez (elips uzunsa).</figcaption></figure>

Şekil 2'deki önemli ayrıntı: elips uzunsa $-\nabla f$ **minimuma doğru bakmaz**; konturlara dik olduğu için dar eksene doğru sapar. Gradyan inişinin dar vadilerde zikzak yapmasının (M1.2) geometrik nedeni budur. Aşağıda kendin dene: noktayı ve yönü değiştir, yönlü türevin $\cos$ eğrisini izle.

::lab l4

### Jacobian: vektör değerli fonksiyonun türevi {#jacobian}

Bir katman bir vektörü başka bir vektöre götürür: $g:\mathbb R^n\to\mathbb R^m$. Her çıktı bileşeninin her girdi bileşenine göre kısmi türevini bir tabloya dizeriz.

:::tanım Jacobian matrisi
$$J_g(x)=\begin{bmatrix}\frac{\partial g_1}{\partial x_1}&\cdots&\frac{\partial g_1}{\partial x_n}\\\vdots&&\vdots\\\frac{\partial g_m}{\partial x_1}&\cdots&\frac{\partial g_m}{\partial x_n}\end{bmatrix}\in\mathbb R^{m\times n},\qquad J_{ij}=\frac{\partial g_i}{\partial x_j}.$$
**Satır** $i$: $g_i$'nin gradyanı. **Sütun** $j$: yalnız $x_j$'yi biraz değiştirince tüm çıktının değişim yönü. Shape (çıktı boyu × girdi boyu). Skaler fonksiyonun ($m=1$) Jacobian'ı $1\times n$ bir satırdır — gradyanın transpozu.
:::

Jacobian'ın anlamı yine doğrusal yaklaşımdır: $g(x+\delta)\approx g(x)+J_g(x)\,\delta$. Yani $x$ yakınında doğrusal olmayan $g$, **matrisi $J$ olan doğrusal bir dönüşüme** (M0.2) benzer. Küçük bir karenin görüntüsü yaklaşık olarak $J$'nin gerdiği paralelkenardır ve alan oranı $|\det J|$'dir. Aşağıda bunu gör: $\varepsilon$'u küçült, yakınlaştırmayı aç — eğri görüntü ile paralelkenar çakışır.

::lab l3

**Derin öğrenmede sık görülen Jacobian'lar:**

| işlem | Jacobian | yorum |
|---|---|---|
| $y=Wx$ | $W$ | doğrusal katmanın Jacobian'ı kendisidir |
| $y=x+b$ | $I$ | toplama gradyanı olduğu gibi geçirir |
| $y=\phi(x)$ eleman eleman (ReLU, σ, tanh) | $\mathrm{diag}(\phi'(x))$ | köşegen: her çıktı yalnız kendi girdisine bağlı |
| $y=\mathrm{softmax}(x)$ | $\mathrm{diag}(y)-yy^\top$ | tam dolu: her çıktı tüm girdilere bağlı (toplam 1) |
| $y=x\odot s$ | $\mathrm{diag}(s)$ | ölçekleme, dropout maskesi |

:::türetim Softmax Jacobian'ı
$y_i=e^{x_i}/S$, $S=\sum_ke^{x_k}$. Bölüm kuralı: $\frac{\partial y_i}{\partial x_j}=\frac{e^{x_i}\,[i=j]\,S-e^{x_i}e^{x_j}}{S^2}=y_i[i=j]-y_iy_j$. Matris olarak $J=\mathrm{diag}(y)-yy^\top$. Her satırın toplamı $y_i-y_i\sum_jy_j=0$: girdilerin hepsine aynı sabiti eklemek softmax'ı değiştirmez (M2.3'teki sayısal kararlılık hilesinin türevdeki karşılığı).
:::

**Pratikte Jacobian neredeyse hiç kurulmaz.** (B, 4096) bir aktivasyonun eleman eleman ReLU Jacobian'ı $4096\times4096$ olurdu ama köşegendir; geri yayılım onu çarpımla, $g\odot\phi'(x)$ olarak uygular. Kütüphanelerin "Jacobian–vektör çarpımı" ile çalışmasının sebebi budur (aşağıda).

### Zincir kuralı {#zincir}

:::sezgi
Dişliler: A dişlisi bir tur dönünce B 3 tur, B bir tur dönünce C 2 tur dönüyorsa, A bir tur dönünce C $3\times2=6$ tur döner. Bileşik sistemde değişim oranları **çarpılır**.
:::

:::tanım Zincir kuralı (tek değişken)
$y=g(x)$, $z=f(y)$, $g$ $x$'te ve $f$ $g(x)$'te türevlenebilir ise
$$\frac{dz}{dx}=\frac{dz}{dy}\cdot\frac{dy}{dx}=f'(g(x))\,g'(x).$$
:::

:::kanıt Doğrusal yaklaşımlardan
$g(x+h)=g(x)+g'(x)h+o(h)$; buna $k=g'(x)h+o(h)$ diyelim. $f(g(x)+k)=f(g(x))+f'(g(x))k+o(k)$. Yerine koyunca $f(g(x+h))=f(g(x))+f'(g(x))g'(x)h+\underbrace{f'(g(x))\,o(h)+o(k)}_{o(h)}$. ($k=O(h)$ olduğu için $o(k)=o(h)$.) Doğrusal terimin katsayısı türevdir: $f'(g(x))g'(x)$. **Sezgisel özet:** iki doğrusal yaklaşımın bileşkesi, eğimlerin çarpımı olan doğrusal yaklaşımdır. ("$dy$'ler sadeleşir" ifadesi akılda tutmak için iyidir ama kanıt değildir.)
:::

**Çok değişkenli zincir kuralı.** $z$, $x$'e birden çok ara değişken üzerinden bağlıysa: $z=f(y_1,\dots,y_k)$, $y_j=g_j(x)$,
$$\frac{\partial z}{\partial x}=\sum_{j=1}^k\frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x}.$$
**Her yolun katkısı toplanır.** Bu toplam, geri yayılımda "bir değişken birden fazla yerde kullanılırsa gradyanları toplanır" kuralının kaynağıdır.

**Matris biçimi.** $h=f\circ g$, $g:\mathbb R^n\to\mathbb R^k$, $f:\mathbb R^k\to\mathbb R^m$ ise
$$J_h(x)=J_f(g(x))\;J_g(x)\qquad (m\times n)=(m\times k)(k\times n).$$
Bileşke dönüşümün matrisi, matrislerin çarpımıdır — M0.2'deki "matris çarpımı = bileşke" ile aynı gerçek, bu kez yerel doğrusal yaklaşımlar için. Yukarıdaki toplam formülü bu çarpımın tek bir elemanıdır.

:::örnek Kutupsal koordinatlar
$x=r\cos\theta$, $y=r\sin\theta$ ve $f(x,y)=x^2+y^2$. Zincirle: $\frac{\partial f}{\partial r}=\frac{\partial f}{\partial x}\cos\theta+\frac{\partial f}{\partial y}\sin\theta=2x\cos\theta+2y\sin\theta=2r(\cos^2\theta+\sin^2\theta)=2r$. Doğrudan: $f=r^2$, $\partial f/\partial r=2r$ ✓. İki yolun katkısı toplandı.
:::

### Hesap grafiği ve geri yayılım {#graf}

Bir ifadeyi her düğümü basit bir işlem olan bir **yönlü döngüsüz grafik** olarak yazalım. $f(x,y)=(x+y)\cdot y$: girdiler $x,y$; ara düğüm $a=x+y$; çıktı $f=a\cdot y$.

<figure class="fig wide"><svg viewBox="0 0 640 200" role="img" aria-label="f = (x+y)·y hesap grafiği: ileri değerler ve geri gradyanlar"><defs><marker id="md3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="20" y="20" width="110" height="54" rx="7" class="mute"/><text x="32" y="42">x</text><text x="32" y="62" class="sm tp">= 2</text><text x="64" y="62" class="sm tb">∂f/∂x = 3</text><rect x="20" y="130" width="110" height="54" rx="7" class="mute"/><text x="32" y="152">y</text><text x="32" y="172" class="sm tp">= 3</text><text x="64" y="172" class="sm tb">∂f/∂y = 8</text><rect x="250" y="55" width="130" height="54" rx="7" class="mute"/><text x="262" y="77">a = x + y</text><text x="262" y="97" class="sm tp">= 5</text><text x="300" y="97" class="sm tb">∂f/∂a = 3</text><rect x="480" y="90" width="130" height="54" rx="7" class="mute"/><text x="492" y="112">f = a · y</text><text x="492" y="132" class="sm tp">= 15</text><text x="535" y="132" class="sm tb">∂f/∂f = 1</text><line x1="130" y1="47" x2="248" y2="75" class="ink" marker-end="url(#md3)"/><line x1="130" y1="157" x2="248" y2="92" class="ink" marker-end="url(#md3)"/><line x1="380" y1="82" x2="478" y2="110" class="ink" marker-end="url(#md3)"/><line x1="130" y1="165" x2="478" y2="128" class="ink" marker-end="url(#md3)"/><text x="160" y="52" class="sm tb">×1</text><text x="160" y="125" class="sm tb">×1</text><text x="410" y="86" class="sm tb">× y = 3</text><text x="300" y="160" class="sm tb">× a = 5 (doğrudan yol)</text></svg><figcaption>Şekil 3 — Yeşil: ileri geçişte hesaplanan değerler. Turuncu: geri geçişte gelen gradyanlar ve kenarlardaki yerel türevler. y iki yoldan kullanıldığı için 3·1 + 5 = 8.</figcaption></figure>

**İleri geçiş:** girdilerden çıktıya, her düğümün değeri hesaplanır ve **saklanır** (geri geçişte yerel türevler bu değerlere ihtiyaç duyar).

**Geri geçiş:** çıktıdan girdilere doğru, her düğüm için
$$\underbrace{\frac{\partial f}{\partial \text{girdi}}}_{\text{aşağı giden}}\;\mathrel{+}=\;\underbrace{\frac{\partial f}{\partial \text{düğüm}}}_{\text{yukarıdan gelen}}\cdot\underbrace{\frac{\partial\,\text{düğüm}}{\partial\,\text{girdi}}}_{\text{yerel türev}}.$$
Her düğüm yalnız **kendi** yerel türevini bilmek zorundadır; grafiğin geri kalanını bilmesi gerekmez. Bu yerellik, otomatik türevi modüler yapar: yeni bir işlem eklemek için yalnız onun ileri ve yerel türev fonksiyonunu yazarsın (M3.1'de kendi autograd motorunu yazacaksın).

Düğümlerin işlenme sırası **ters topolojik sıra**dır: bir düğümün gradyanı, onu kullanan **tüm** düğümler işlendikten sonra tamamlanmış olur. Şekil 3'te $y$'nin gradyanı iki adımda birikir: önce $f$ düğümünden doğrudan $5$, sonra $a$ düğümünden $3$.

Aşağıda üç grafikte adım adım ilerle: ifadeyi seç, girdileri değiştir, "İleri" ile her düğümün hesabını gör. En sonda geri yayılımın sonucu sayısal türevle karşılaştırılır.

::lab l2

**Kapı kalıpları** (CS231n'in sezgileri) — geri geçişte işlemler gradyana ne yapar:

| düğüm | yerel türevler | gradyana etkisi |
|---|---|---|
| toplama $a+b$ | $1,\ 1$ | **dağıtıcı:** gelen gradyanı değiştirmeden her girdiye kopyalar |
| çarpma $a\cdot b$ | $b,\ a$ | **takas:** her girdiye gelen gradyan × **diğer** girdinin değeri; büyük girdi, karşısındakine büyük gradyan verir |
| $\max(a,b)$, ReLU | $[a>b],\ [b\ge a]$ | **yönlendirici:** gradyanın tamamı kazanana gider, kaybeden 0 alır |
| kopya / dallanma | — | **toplayıcı:** bir değer birden çok yerde kullanılırsa gelen gradyanlar toplanır |
| sigmoid, tanh | $\sigma(1-\sigma)$, $1-\tanh^2$ | **kısıcı:** doymuş bölgede gradyanı neredeyse sıfıra indirir |

:::yanılgı "Geri geçiş türevleri sembolik olarak sadeleştirir"
Hayır. Geri yayılım (ters mod otomatik türev) türevin **formülünü** değil, verilen noktadaki **sayısal değerini** hesaplar. Sembolik türev (Mathematica, SymPy) formül üretir ve ifade büyüdükçe formül patlayabilir; sayısal türev (sonlu farklar) yaklaşıktır ve yavaştır. Otomatik türev ikisinin arasındadır: kesin (yuvarlama hatası dışında) ve hızlı, ama yalnız bir noktada.
:::

### Tam bir örnek: tek nöron ve kare kayıp {#noron}

$z=wx+b$, $a=\sigma(z)$, $L=(a-t)^2$. Değerler: $w=0.8$, $x=1.5$, $b=-0.5$, $t=1$ (laboratuvardaki "Nöron" ifadesiyle aynı).

**İleri:** $z=0.8\cdot1.5-0.5=0.7$; $a=\sigma(0.7)=0.6682$; $e=a-t=-0.3318$; $L=e^2=0.1101$.

**Geri:**

1. $\partial L/\partial L=1$.
2. $L=e^2$: $\partial L/\partial e=2e=-0.6637$.
3. $e=a-t$: $\partial L/\partial a=-0.6637\cdot1$, $\partial L/\partial t=-0.6637\cdot(-1)=0.6637$.
4. $a=\sigma(z)$: yerel $\sigma(z)(1-\sigma(z))=0.6682\cdot0.3318=0.2217$; $\partial L/\partial z=-0.6637\cdot0.2217=-0.1471$.
5. $z=wx+b$: $\partial L/\partial w=-0.1471\cdot x=-0.2207$; $\partial L/\partial x=-0.1471\cdot w=-0.1177$; $\partial L/\partial b=-0.1471$.

**Yorum:** $\partial L/\partial w<0$, yani $w$'yi artırmak kaybı azaltır: $a$ hedefin (1) altında, $w$ artınca $z$ ve $a$ artar. Gradyan inişi tam olarak bunu yapar: $w\leftarrow w-\eta\cdot(-0.2207)$. $\partial L/\partial b=\partial L/\partial z$ ve $\partial L/\partial w=\partial L/\partial z\cdot x$: ağırlığın gradyanı, girdinin değeriyle ölçeklenir. **Girdi sıfıra yakınsa o ağırlık neredeyse hiç öğrenmez** (Toygar BİL 442 Alıştırma 2'deki soru); bias ise girdiye bağlı değildir.

### İleri mod ve ters mod {#modlar}

Zincir kuralındaki Jacobian çarpımını **hangi sırayla** yaptığımız maliyeti belirler. $L=f_3(f_2(f_1(x)))$, $x\in\mathbb R^n$, $L\in\mathbb R$:
$$\nabla_xL^\top=\underbrace{J_3}_{1\times k_2}\;\underbrace{J_2}_{k_2\times k_1}\;\underbrace{J_1}_{k_1\times n}.$$

- **İleri mod (sağdan sola, girdiden çıktıya):** bir girdi yönü $v$ seç, $J_1v$, sonra $J_2(J_1v)$, … hesapla: **Jacobian–vektör çarpımı (JVP)**. Bir geçiş bir yönlü türevi verir; tam gradyan için $n$ geçiş gerekir.
- **Ters mod (soldan sağa, çıktıdan girdiye):** $u^\top=1$ ile başla, $u^\top J_3$, sonra $(u^\top J_3)J_2$, …: **vektör–Jacobian çarpımı (VJP)**. Bir geçiş bütün gradyanı verir — geri yayılım budur.

| | ileri mod (JVP) | ters mod (VJP) |
|---|---|---|
| bir geçişte | bir girdi yönüne göre tüm çıktıların türevi (J'nin bir sütunu) | bir çıktının tüm girdilere göre türevi (J'nin bir satırı) |
| verimli olduğu durum | az girdi, çok çıktı ($n\ll m$) | çok girdi, az çıktı ($n\gg m$) — eğitim: milyonlarca parametre, 1 kayıp |
| bellek | ara değerleri saklamaya gerek yok | ileri geçişin ara değerleri saklanmalı |
| PyTorch | `torch.func.jvp`, `jacfwd` | `.backward()`, `torch.func.vjp`, `grad`, `jacrev` |

**Ucuz gradyan ilkesi.** Ters modla skaler bir fonksiyonun gradyanını hesaplamak, fonksiyonu hesaplamanın **sabit bir katı** (kuramsal üst sınır ~4–5, pratikte 2–3) kadar sürer — parametre sayısından bağımsız olarak (Griewank ve Walther). Sayısal türev ise $2n$ fonksiyon değerlendirmesi ister: $n=10^8$ parametrede ters mod $\sim3$ ileri geçiş, sonlu fark $2\cdot10^8$ ileri geçiş.

**Bedeli bellek:** ters mod ileri geçişin ara değerlerini saklamak zorundadır. Büyük modellerde eğitim belleğinin büyük kısmı ağırlıklar değil **aktivasyonlardır**; gradient checkpointing (M7) bazı ara değerleri atıp geri geçişte yeniden hesaplayarak bellek–hesap takası yapar.

:::ayrıntı İleri mod nasıl uygulanır: dual sayılar
Her sayıyı $(v,\dot v)$ çifti olarak taşı: değer ve türev. Kurallar: $(a,\dot a)+(b,\dot b)=(a+b,\dot a+\dot b)$, $(a,\dot a)(b,\dot b)=(ab,\dot ab+a\dot b)$, $f((a,\dot a))=(f(a),f'(a)\dot a)$. Girdiyi $(x,1)$ ile başlat; çıktının ikinci bileşeni $f'(x)$'tir. Saklama gerekmez, ama her girdi yönü için ayrı geçiş gerekir.
:::

**Tarihçe.** Ters mod otomatik türev ilk kez Seppo Linnainmaa'nın 1970 yüksek lisans tezinde yayımlandı; Werbos (1974) sinir ağlarına uygulanabileceğini önerdi; Rumelhart, Hinton ve Williams (1986) çok katmanlı ağların bu yolla eğitilebileceğini gösterdi ve yöntem "backpropagation" adıyla yayıldı.

### Sayısal türev ve gradyan kontrolü {#sayisal}

Türevi tanımdaki limiti almadan, küçük bir $h$ ile yaklaşık hesaplayabiliriz. Kodda yazdığın geri geçişi doğrulamak için vazgeçilmezdir.

:::türetim İleri farkın ve merkezi farkın hatası
Taylor: $f(x\pm h)=f(x)\pm f'(x)h+\tfrac12f''(x)h^2\pm\tfrac16f'''(x)h^3+O(h^4)$.

**İleri fark:** $\frac{f(x+h)-f(x)}{h}=f'(x)+\tfrac12f''(x)h+O(h^2)$. Hata $O(h)$: $h$ 10 kat küçülünce hata ~10 kat küçülür.

**Merkezi fark:** iki açılımı çıkar — çift kuvvetler sadeleşir: $f(x+h)-f(x-h)=2f'(x)h+\tfrac13f'''(x)h^3+O(h^5)$, böylece $\frac{f(x+h)-f(x-h)}{2h}=f'(x)+\tfrac16f'''(x)h^2+O(h^4)$. Hata $O(h^2)$: $h$ 10 kat küçülünce hata ~100 kat küçülür. Bu bir **eşitsizlik değil asimptotik** ifadedir: $f'''$ büyükse sabit de büyüktür.
:::

**Yuvarlama hatası** ters yönde çalışır. Kayan noktalı sayılarda her değer göreli $\varepsilon$ kadar hatalıdır (float64'te $\varepsilon\approx2.2\cdot10^{-16}$, float32'de $1.2\cdot10^{-7}$). $f(x+h)$ ile $f(x)$ çok yakınsa farkları **ardışık iptal** (catastrophic cancellation) yüzünden göreli olarak çok hatalıdır: farktaki mutlak hata $\sim\varepsilon|f|$, bunu $h$'ye bölünce $\sim\varepsilon|f|/h$. Toplam hata kabaca
$$\text{ileri: } \tfrac12|f''|h+\frac{\varepsilon|f|}{h}\ \Rightarrow\ h^*\sim\sqrt\varepsilon,\qquad \text{merkezi: } \tfrac16|f'''|h^2+\frac{\varepsilon|f|}{h}\ \Rightarrow\ h^*\sim\varepsilon^{1/3}.$$
float64'te ileri fark için $h\approx10^{-8}$, merkezi fark için $h\approx10^{-5}$ makul seçimlerdir. Yukarıdaki ilk deneyde hata eğrisinin V biçimini gör; float32'ye geçince V'nin dibinin ne kadar yükseldiğine dikkat et.

**Gradyan kontrolü — uygulamada kurallar:**

1. Merkezi farkı **float64**'te kullan; float32'de yalnız 2–3 basamak güvenilir.
2. **Göreli hata** ile karşılaştır: $\frac{|g_a-g_n|}{\max(|g_a|,|g_n|,\epsilon)}$. Kabaca $<10^{-7}$ iyi, $10^{-4}$ civarı şüpheli (kırılma noktası olabilir), $>10^{-2}$ neredeyse kesin hata.
3. **Kırılma noktaları** (ReLU, max, |x|) yanlış alarm verir: $x\pm h$ farklı taraflara düşerse merkezi fark iki eğimin ortalamasını verir. Az sayıda koordinatı rastgele kontrol et ve sonuçları yorumla.
4. Dropout ve rastgele veri artırma gibi rastgele kısımları **kapat** (ya da tohumu sabitle), yoksa $f(x+h)$ ve $f(x-h)$ farklı fonksiyonlardan gelir.
5. Düzenlileştirme terimini ayrı kontrol et: büyük olursa veri teriminin gradyan hatasını gizleyebilir.

PyTorch bunun için `torch.autograd.gradcheck` sunar (float64 girdiler bekler).

### Türevin olmadığı yerler: alt-gradyan {#altgradyan}

ReLU $x=0$'da türevlenebilir değildir ama ağlar sorunsuz eğitilir. Neden?

- **Olasılık sıfır:** sürekli rastgele başlatma ve verilerle bir ön-aktivasyonun **tam** sıfır olması pratikte çok nadirdir (float hassasiyetinde ara sıra olur).
- **Alt-gradyan:** dışbükey bir $f$ için $g$, $x$'te bir alt-gradyandır ancak ve ancak her $y$ için $f(y)\ge f(x)+g(y-x)$ (fonksiyon, o eğimdeki doğrunun hep üstünde). $|x|$'in $0$'daki alt-gradyanları $[-1,1]$ aralığıdır; ReLU'nunki $[0,1]$. Çerçeveler bu aralıktan **bir değer seçer**: PyTorch ReLU için $0$ kullanır.
- **Dikkat:** "alt-gradyan" kavramı dışbükey fonksiyonlar içindir; sinir ağının bütünü dışbükey değildir. Sonuç kuramsal bir garanti değil, pratikte işe yarayan bir **uzlaşıdır**. Bu seçim nadiren önemlidir ama, örneğin, `max` ile eşit iki değerin hangisinin gradyan alacağı kütüphaneye göre değişebilir.

### PyTorch'ta otomatik türev {#pytorch}

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
f = (x + y) * y      # ileri geçiş: grafik kaydedilir (f.grad_fn = MulBackward0)
f.backward()         # ters topolojik sırayla geri geçiş
print(x.grad, y.grad)  # tensor(3.) tensor(8.)
```

**Bilinmesi gereken davranışlar:**

- `requires_grad=True` olan tensörlerle yapılan her işlem bir düğüm olarak kaydedilir (`grad_fn`). Yapraklar (kullanıcının oluşturduğu tensörler) `.grad` biriktirir; ara tensörler varsayılan olarak biriktirmez (`retain_grad()` ile isteyebilirsin).
- **`.grad` toplanır, üzerine yazılmaz.** İkinci bir `backward()` gradyanı ekler. Eğitim döngüsünde her adımda `optimizer.zero_grad()` bu yüzden gerekir (ya da bilerek gradyan biriktirme için atlanır).
- `backward()` grafiği varsayılan olarak **serbest bırakır** (ara değerlerin belleği geri verilir); aynı grafikte iki kez geri gitmek için `retain_graph=True`.
- Skaler olmayan bir çıktıda `backward(v)` bir vektör ister: VJP $v^\top J$'yi hesaplar. `y.sum().backward()`, $v=\mathbf 1$ ile aynıdır.
- `torch.no_grad()` / `torch.inference_mode()` grafik kaydını kapatır (çıkarımda bellek ve hız); `x.detach()` gradyan akışını o noktada keser.

**`torch.func`** (fonksiyonel API, JAX benzeri):

| fonksiyon | ne döndürür |
|---|---|
| `grad(f)(x)` | skaler $f$'nin gradyanı |
| `vjp(f, x)` | $f(x)$ ve $v\mapsto v^\top J$ fonksiyonu |
| `jvp(f, (x,), (v,))` | $f(x)$ ve $Jv$ |
| `jacrev(f)(x)`, `jacfwd(f)(x)` | tam Jacobian (ters / ileri mod) |
| `hessian(f)(x)` | ikinci türevler matrisi ($=$ `jacfwd(jacrev(f))`) |
| `vmap(F)(X)` | tek örnek için yazılmış `F`'yi batch boyutu üzerinde döngüsüz çalıştırır |

Modülün **e** aşamasında `vmap(grad(f))` ile yüzlerce noktanın gradyanını tek çağrıda alacaksın. Örnek başına gradyan (diferansiyel gizlilik), Jacobian izleri ve Hessian–vektör çarpımları bu araçlarla yazılır.

:::ayrıntı İkinci türev: Hessian
$H_{ij}=\frac{\partial^2f}{\partial x_i\partial x_j}$ ($n\times n$, sürekli ikinci türevlerde simetrik). Doğrusal yaklaşımın bir sonraki terimi: $f(x+\delta)\approx f(x)+\nabla f^\top\delta+\tfrac12\delta^\top H\delta$. Özdeğerleri eğriliği verir: hepsi pozitifse yerel minimum, karışık işaretliyse eyer noktası. Büyük ağlarda $H$ kurulmaz ($n^2$ eleman); $Hv$ çarpımı iki geri geçişle (gradyanın $v$ yönündeki türevi) hesaplanır ve optimizasyon analizinde (en büyük özdeğer → kararlı öğrenme oranı sınırı, M1.2) kullanılır.
:::

### Özet ve sembol tablosu {#ozet}

| sembol | anlamı | shape (girdi $\mathbb R^n$, çıktı $\mathbb R^m$) |
|---|---|---|
| $f'(x)$, $\frac{df}{dx}$ | tek değişkenli türev | skaler |
| $\frac{\partial f}{\partial x_i}$ | kısmi türev: diğerleri sabit | skaler |
| $\nabla f$ | gradyan ($m=1$) | girdiyle aynı: $n$ |
| $D_uf=\nabla f\cdot u$ | $u$ yönünde yönlü türev | skaler |
| $J_g$ | Jacobian, $J_{ij}=\partial g_i/\partial x_j$ | $m\times n$ |
| $H$ | Hessian | $n\times n$ |
| JVP / VJP | $Jv$ (ileri mod) / $v^\top J$ (ters mod) | $m$ / $n$ |

**Akılda kalsın:** türev = en iyi doğrusal yaklaşım. Zincir kuralı = doğrusal yaklaşımların bileşkesi = Jacobian'ların çarpımı. Geri yayılım bu çarpımı çıktı tarafından başlayarak yapar, her düğümde "yukarıdan gelen × yerel türev" hesaplar ve birden çok yoldan gelen katkıları toplar; tek bir geri geçiş tüm parametrelerin gradyanını verir. Yazdığın her geri geçişi merkezi farkla, float64'te, göreli hatayla doğrula.

**Kaynaklar:** CS231n, "Backpropagation, Intuitions" (kapı kalıpları, hesap grafiği); Deisenroth, Faisal ve Ong, *Mathematics for Machine Learning*, Bölüm 5 (kısmi türev, Jacobian, zincir kuralı, otomatik türev); Baydin ve ark. (2018), "Automatic differentiation in machine learning: a survey", JMLR (ileri/ters mod, dual sayılar); Griewank ve Walther, *Evaluating Derivatives* (ucuz gradyan ilkesi); Goodfellow ve ark., *Deep Learning* Bölüm 6.5; Karpathy, micrograd; PyTorch `torch.func` ve autograd belgeleri; Toygar, BİL 442 Bölüm 10 slaytları (backprop akışı, tarihçe) ve Alıştırma 2.
