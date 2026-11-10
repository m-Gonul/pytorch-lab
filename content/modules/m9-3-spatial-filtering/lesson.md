### Uzamsal süzgeçleme {#temel}

Uzamsal süzgeçleme görüntü işlemenin çok geniş bir yelpazesinde kullanılır. "Süzgeç" adı frekans alanından gelir: orada süzgeçleme, görüntünün belirli frekans bileşenlerini geçirmek, değiştirmek ya da reddetmek demektir. Düşük frekansları geçiren **alçak geçiren** bir süzgeç görüntüyü bulanıklaştırarak yumuşatır; aynı yumuşatmayı doğrudan görüntü üzerinde uzamsal süzgeçlerle yapabiliriz (frekans tarafı M9.10'da).

Uzamsal süzgeçleme, her pikselin değerini o pikselin ve komşularının değerlerinin bir fonksiyonuyla değiştirir. Pikseller üzerindeki işlem doğrusalsa süzgece **doğrusal uzamsal süzgeç**, değilse **doğrusal olmayan** süzgeç denir.

### Doğrusal süzgeçlemenin mekaniği {#mekanik}

Doğrusal bir uzamsal süzgeç, bir $f$ görüntüsü ile bir **süzgeç çekirdeği** $w$ arasında **çarpımlar toplamı** işlemi yapar. Çekirdeğin boyutu işlem komşuluğunu, katsayıları süzgecin doğasını belirler. Çekirdek için maske, şablon ve pencere terimleri de kullanılır. 3 × 3'lük bir çekirdekle herhangi bir $(x,y)$ noktasındaki yanıt, çekirdek katsayıları ile kapsadığı piksellerin çarpımlarının toplamıdır:
$$g(x,y)=w(-1,-1)f(x-1,y-1)+w(-1,0)f(x-1,y)+\dots+w(0,0)f(x,y)+\dots+w(1,1)f(x+1,y+1).$$
Merkez katsayı $w(0,0)$, $(x,y)$'deki pikselle hizalanır. $m\times n$'lik bir çekirdekte $m=2a+1$, $n=2b+1$ (iki yönde de tek boyut) varsayılır. Genel hâl:
$$g(x,y)=\sum_{s=-a}^a\sum_{t=-b}^bw(s,t)\,f(x+s,y+t).$$

### Korelasyon ve konvolüsyon {#korelasyon}

Yukarıdaki formül **uzamsal korelasyonun** tanımıdır: çekirdeğin merkezi görüntü üzerinde kaydırılır ve her konumda çarpımlar toplamı alınır. **Uzamsal konvolüsyonun** mekaniği aynıdır, tek fark çekirdeğin **180° döndürülmesidir**:
$$(w\star f)(x,y)=\sum_{s,t}w(s,t)f(x+s,y+t),\qquad(w\ast f)(x,y)=\sum_{s,t}w(s,t)f(x-s,y-t)=\sum_{s',t'}w(-s',-t')f(x+s',y+t').$$
Çekirdek merkezine göre simetrikse ikisi aynı sonucu verir.

<figure class="fig wide"><svg viewBox="0 0 640 156" role="img" aria-label="Korelasyon ve konvolüsyon: birim dürtüde korelasyon çekirdeği ters çevrilmiş, konvolüsyon aynen verir; 3×3 parçada köşe çekirdeğiyle korelasyon 1, konvolüsyon 9 verir"><text x="10" y="16" class="sm">1-B: f bir birim dürtü, w = [1, 2, 3]</text><text x="10" y="46" class="sm">f</text><rect x="60" y="30" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="74.0" y="46" text-anchor="middle" class="sm">0</text><rect x="90" y="30" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="104.0" y="46" text-anchor="middle" class="sm">0</text><rect x="120" y="30" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="134.0" y="46" text-anchor="middle" class="sm">0</text><rect x="150" y="30" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="164.0" y="46" text-anchor="middle" class="sm">1</text><rect x="180" y="30" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="194.0" y="46" text-anchor="middle" class="sm">0</text><rect x="210" y="30" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="224.0" y="46" text-anchor="middle" class="sm">0</text><rect x="240" y="30" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="254.0" y="46" text-anchor="middle" class="sm">0</text><text x="10" y="80" class="ta">korelasyon</text><rect x="120" y="64" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="134.0" y="80" text-anchor="middle" class="sm">0</text><rect x="150" y="64" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="164.0" y="80" text-anchor="middle" class="sm">3</text><rect x="180" y="64" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="194.0" y="80" text-anchor="middle" class="sm">2</text><rect x="210" y="64" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="224.0" y="80" text-anchor="middle" class="sm">1</text><rect x="240" y="64" width="28" height="24" class="soft-a" stroke="var(--line)"/><text x="254.0" y="80" text-anchor="middle" class="sm">0</text><text x="10" y="114" class="tb">konvolüsyon</text><rect x="120" y="98" width="28" height="24" class="soft-b" stroke="var(--line)"/><text x="134.0" y="114" text-anchor="middle" class="sm">0</text><rect x="150" y="98" width="28" height="24" class="soft-b" stroke="var(--line)"/><text x="164.0" y="114" text-anchor="middle" class="sm">1</text><rect x="180" y="98" width="28" height="24" class="soft-b" stroke="var(--line)"/><text x="194.0" y="114" text-anchor="middle" class="sm">2</text><rect x="210" y="98" width="28" height="24" class="soft-b" stroke="var(--line)"/><text x="224.0" y="114" text-anchor="middle" class="sm">3</text><rect x="240" y="98" width="28" height="24" class="soft-b" stroke="var(--line)"/><text x="254.0" y="114" text-anchor="middle" class="sm">0</text><text x="10" y="146" class="sm">Dürtüde korelasyon döndürülmüş çekirdeği, konvolüsyon tam kopyasını verir.</text><text x="330" y="16" class="sm">2-B: görüntü parçası ve köşe çekirdeği</text><rect x="330" y="28" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="344" y="44" text-anchor="middle" class="sm">1</text><rect x="360" y="28" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="374" y="44" text-anchor="middle" class="sm">2</text><rect x="390" y="28" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="404" y="44" text-anchor="middle" class="sm">3</text><rect x="330" y="54" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="344" y="70" text-anchor="middle" class="sm">4</text><rect x="360" y="54" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="374" y="70" text-anchor="middle" class="sm">5</text><rect x="390" y="54" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="404" y="70" text-anchor="middle" class="sm">6</text><rect x="330" y="80" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="344" y="96" text-anchor="middle" class="sm">7</text><rect x="360" y="80" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="374" y="96" text-anchor="middle" class="sm">8</text><rect x="390" y="80" width="28" height="24" class="soft-p" stroke="var(--line)"/><text x="404" y="96" text-anchor="middle" class="sm">9</text><text x="430" y="44" class="sm">w = [[1,0,0],[0,0,0],[0,0,0]]</text><text x="430" y="70" class="ta">korelasyon = 1</text><text x="430" y="94" class="tb">konvolüsyon = 9</text><text x="430" y="118" class="sm">Laplasyen: 2+4+6+8−20 = 0</text></svg><figcaption>Şekil 1 — Korelasyon ve konvolüsyonun tek farkı çekirdeğin 180° döndürülmesidir; simetrik çekirdeklerde ikisi aynıdır. Sağda modülün el hesapları.</figcaption></figure>

**Dürtü yanıtı.** Tek bir 1 ve geri kalanı 0 olan fonksiyona **ayrık birim dürtü** denir. Bir çekirdeği dürtüyle **korelasyona** sokmak, dürtünün konumunda çekirdeğin 180° döndürülmüş hâlini verir. **Konvolüsyonda** ise çekirdek önceden döndürüldüğü için dürtünün konumunda çekirdeğin tam bir kopyası elde edilir. Doğrusal sistem kuramının bir temeli de budur: bir fonksiyonu dürtüyle konvolüsyona sokmak, fonksiyonun dürtünün konumunda bir kopyasını verir. Konvolüsyonun değişme ve birleşme özellikleri ($w_1\ast(w_2\ast f)=(w_1\ast w_2)\ast f$) de bu döndürme sayesinde vardır; korelasyonda yoktur.

**El hesabı.** Görüntü parçası $\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}$, merkez pikselde. Sol üst köşesi 1 olan çekirdekle **korelasyon** $1\cdot f(x-1,y-1)=\mathbf1$; **konvolüsyon** (çekirdek döner, 1 sağ alta geçer) $\mathbf9$. Laplasyen çekirdeği $\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$ ile: $2+4+6+8-4\cdot5=\mathbf0$ (yerel olarak doğrusal bir rampada ikinci türev sıfırdır; M9.4).

:::uyarı Terim karmaşası
Doğrusal uzamsal süzgeçleme ile uzamsal konvolüsyon çoğu zaman eş anlamlı kullanılır. "Konvolüsyon süzgeci", "konvolüsyon maskesi", "konvolüsyon çekirdeği" terimleri bir süzgeç çekirdeğini anlatır, çekirdeğin mutlaka konvolüsyonla kullanıldığını değil. "Bir çekirdeği görüntüyle konvolüsyona sokmak" genellikle kaydırma ve çarpımlar toplamı sürecini anlatır, korelasyonla konvolüsyonu ayırmaz. `cv2.filter2D` ve derin öğrenmedeki `Conv2d` gerçekte **korelasyon** yapar (M4.1). Çekirdek öğreniliyorsa fark önemsizdir; elle tasarlanmış asimetrik bir çekirdek (ör. yönlü türev) aktarılırken önemlidir.
:::

### Kenar dolgusu {#dolgu}

1 × 1'den büyük her çekirdekte, görüntünün kenarlarında bazı girdi pikselleri eksiktir. İki seçenek vardır: tam komşuluğu olmayan pikselleri atlamak (çıkış küçülür) ya da girdiyi **doldurmak**. Yaygın dolgu yöntemleri:
1. **Sıfır dolgu**: dışarısı 0 (OpenCV `BORDER_CONSTANT`).
2. **Kenar pikselini tekrarlama**: `aaa|abc` (`BORDER_REPLICATE`).
3. **Yansıtma**: `cba|abc` (`BORDER_REFLECT`) ya da kenar pikseli tekrarlanmadan `dcb|abcd` (`BORDER_REFLECT_101`, OpenCV'nin varsayılanı).
4. **Dairesel**: görüntü periyodikmiş gibi `xyz|abc` (`BORDER_WRAP`; DFT ile süzgeçlemenin örtük varsayımı, M9.10).

Sıfır dolgu kenarları karanlığa çeker; görüntünün kenarlarında koyu bir şerit belirir. Ek x1 bütün modları `cv2.copyMakeBorder` ile karşılaştırıyor.

::lab l2

### Ayrılabilir çekirdekler {#ayrilabilir}

Bir $G(x,y)$ fonksiyonu iki 1-B fonksiyonun çarpımı, $G(x,y)=G_1(x)G_2(y)$ olarak yazılabiliyorsa **ayrılabilirdir**. Bir çekirdek matristir; ayrılabilir çekirdek iki vektörün **dış çarpımıdır**: $\mathbf w=\mathbf c\,\mathbf r^\top$. Örneğin $2\times3$'lük birler matrisi $[1,1]^\top[1,1,1]$'dir.

:::uyarı Kaynak düzeltmesi (slayt 69)
Slayt tanımı $G(x,y)=G_1(x)G_1(x)$ biçiminde yazıyor; doğrusu $G(x,y)=G_1(x)\,G_2(y)$: biri yalnız $x$'e, diğeri yalnız $y$'ye bağlı iki fonksiyonun çarpımı.
:::

Bir sütun vektörü ile bir satır vektörünün çarpımı, iki vektörün 2-B konvolüsyonuyla aynıdır. Bu yüzden ayrılabilir bir çekirdekle konvolüsyon, iki 1-B konvolüsyonla (konvolüsyon değişmeli ve birleşmeli olduğu için herhangi bir sırada) yapılabilir. Bir sütun vektörü ile bir satır vektörünün çarpımı her zaman **rank 1**'dir; bir çekirdeğin ayrılabilir olup olmadığını anlamak için rankının 1 olup olmadığına bakmak yeter (pratikte SVD ile; ek x2).

:::türetim Ayrılabilirliğin kazancı
$M\times N$ bir görüntü ve $m\times n$ bir çekirdekle süzgeçleme ≈ $MNmn$ çarpma ve toplama ister. Çekirdek ayrılabilirse ilk konvolüsyon ($m\times1$) $MNm$, ikincisi ($1\times n$) $MNn$ işlem ister; toplam $MN(m+n)$. Kazanç
$$C=\frac{MNmn}{MN(m+n)}=\frac{mn}{m+n}.$$
El hesabı: $\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}=[1,2,1]^\top[1,2,1]$: doğrudan piksel başına $\mathbf9$ çarpma, ayrılmış $\mathbf6$.
:::

:::uyarı Kaynak düzeltmesi: 11 × 11'lik kazanç (slayt 72)
Slayt (Gonzalez–Woods'u izleyerek) 11 × 11'lik bir çekirdek için kazancı "5.2" verir. Formülle $C=\frac{11\cdot11}{11+11}=\frac{121}{22}=\mathbf{5.5}$'tir.
:::

### Yumuşatma süzgeçleri {#yumusatma}

Yumuşatma (ortalama alma) süzgeçleri yoğunluktaki keskin geçişleri azaltır. Rastgele gürültü tipik olarak keskin geçişlerden oluştuğu için açık bir uygulaması gürültü azaltmadır. Diğerleri: yeniden örneklemeden önce aliasing'i azaltmak (M9.1), çekirdeğe göre küçük olan "ilgisiz" ayrıntıyı azaltmak, yetersiz yoğunluk seviyesinden doğan yanlış konturları yumuşatmak, histogram işleme ve keskinliği azaltılmış maskeleme gibi başka yöntemlerle birleştirmek.

**Kutu çekirdeği.** En basit ayrılabilir alçak geçiren çekirdek: bütün katsayıları aynı (genellikle 1); 3-B'de bir kutu gibi göründüğü için adını alır. $m\times n$'lik kutu süzgeci $m\times n$'lik birler dizisidir ve önündeki normalizasyon sabiti katsayılar toplamının tersidir. Normalizasyonun (bütün alçak geçiren çekirdeklere uygulanır) iki amacı vardır: sabit yoğunluklu bir bölgenin ortalaması süzgeçten sonra aynı kalmalıdır; ve süzgeç bir yanlılık eklememelidir, yani özgün ve süzülmüş görüntünün piksel toplamları aynı olmalıdır. Bütün satır ve sütunları aynı olduğu için kutu çekirdeğinin rankı 1'dir: ayrılabilirdir.

**Gauss çekirdekleri.** Kutu süzgeçleri hızlı deneyler için uygundur ve çoğu zaman görsel olarak kabul edilebilir sonuç verir. Ama sınırları vardır: örneğin odak dışı bir mercek sıklıkla alçak geçiren bir süzgeçle modellenir ve kutu süzgeci merceklerin bulanıklaştırma özelliğine kötü bir yaklaşımdır. Seçilen çekirdekler **dairesel simetrik** (izotropik; yanıtı yöne bağlı değil) olanlardır:
$$w(s,t)=G(s,t)=K\exp\Big(-\frac{s^2+t^2}{2\sigma^2}\Big)=K\exp\Big(-\frac{r^2}{2\sigma^2}\Big),\qquad r=\sqrt{s^2+t^2}.$$
Bu biçimdeki Gauss çekirdekleri, **hem dairesel simetrik hem ayrılabilir** olan tek çekirdeklerdir: $e^{-(s^2+t^2)/2\sigma^2}=e^{-s^2/2\sigma^2}e^{-t^2/2\sigma^2}$. Böylece kutu süzgecinin hesap avantajını ve görüntü işleme için ideal başka özellikleri (iki Gauss'un konvolüsyonu yine Gauss'tur, varyanslar toplanır; frekans yanıtı da Gauss'tur ve yan lobu yoktur) birlikte taşır. Tek boyutlu çekirdeklerde merkez tamsayı konumda olduğu için bütün $r^2$ değerleri tamsayıdır.

**Çekirdek boyu.** Gauss'un değerleri merkezden $3\sigma$ uzakta ihmal edilebilir küçüklüktedir (≈ %1); çekirdek boyu genellikle $\lceil6\sigma\rceil$'ya en yakın tek sayı seçilir. OpenCV `ksize` verilip `sigma ≤ 0` ise $\sigma=0.3\,((k-1)/2-1)+0.8$ kullanır. El hesabı: $k=7$ için $\sigma=0.3\cdot2+0.8=\mathbf{1.4}$ (küçük boyutlarda sabit binom tablolarını kullanır; aşama c).

### Sıra istatistiği süzgeçleri {#medyan}

**Sıra istatistiği süzgeçleri**, yanıtı süzgecin kapsadığı bölgedeki piksellerin sıralanmasına dayanan doğrusal olmayan uzamsal süzgeçlerdir. Yumuşatma, merkez pikselin değerini sıralama sonucunun belirlediği değerle değiştirerek elde edilir. En bilineni **medyan süzgecidir**: merkez pikselin değerini komşuluğundaki yoğunlukların medyanıyla (merkez dahil) değiştirir. Medyan, değerlerin yarısının ondan küçük ya da eşit, yarısının büyük ya da eşit olduğu değerdir.

**El hesabı (düz ve ağırlıklı medyan).** Komşuluk $\begin{bmatrix}38&17&91\\14&111&5\\29&61&42\end{bmatrix}$. Sıralı: 5, 14, 17, 29, **38**, 42, 61, 91, 111 → medyan **38**. Ağırlıklar $\begin{bmatrix}1&2&1\\2&5&2\\1&2&1\end{bmatrix}$ ile her değer ağırlığı kadar tekrarlanır (toplam 17 eleman): 5, 5, 14, 14, 17, 17, 29, 38, **42**, 61, 61, 91, 111 (beş kez) → 9. eleman **42**. Ağırlıklı medyan merkeze ve yakın komşulara daha çok söz hakkı verir; ince ayrıntıyı biraz daha korur.

Medyan süzgeçleri ("aykırı değer temizleyiciler") belirli gürültü türlerinde, benzer boyutlu doğrusal süzgeçlerden çok daha az bulanıklaştırarak mükemmel gürültü azaltma sağlar. Özellikle **darbe gürültüsünde** (beyaz ve siyah noktalar olarak görüldüğünde tuz-biber gürültüsü) etkilidir: aykırı değerler sıralamanın uçlarına düşer ve hiçbir zaman medyan olmaz. Medyan, sıralı kümenin 50. yüzdeliğidir; başka yüzdelikler de kullanılabilir: 100. yüzdelik **maks süzgecidir** (en parlak noktaları bulmak, açık bölgelere komşu koyu alanları aşındırmak için), 0. yüzdelik **min süzgecidir** (tersi için). Bunlar morfolojik genişletme ve aşındırmanın gri tonlu hâlleridir (M9.8).

::lab l1

### İntegral görüntü ve kayan pencere {#integral}

Kutu süzgecinin maliyeti çekirdek boyuyla büyür (ayrılmış hâlde bile $m+n$). Kutu boyundan **bağımsız** maliyet mümkündür. **İntegral görüntü** (summed-area table; Crow 1984) her konumda sol üstteki bütün piksellerin toplamını saklar: $A(y,x)=\sum_{r\le y,\,c\le x}I(r,c)$. Bir kez $O(MN)$'de hesaplanır. Herhangi bir dikdörtgenin toplamı dört köşe değerinden bulunur.

<figure class="fig wide"><svg viewBox="0 0 640 158" role="img" aria-label="İntegral görüntü: her hücre sol üstündeki bütün piksellerin toplamı; bir dikdörtgenin toplamı dört köşe değerinden hesaplanır"><text x="10" y="16" class="sm">görüntü I</text><text x="200" y="16" class="sm">integral görüntü A(y, x) = Σ<tspan baseline-shift="-3" font-size="75%">r ≤ y, c ≤ x</tspan> I(r, c)</text><rect x="10" y="26" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="25" y="42" text-anchor="middle" class="sm">1</text><rect x="200" y="26" width="38" height="24" class="soft-p" stroke="var(--line)"/><text x="219" y="42" text-anchor="middle" class="sm">1</text><rect x="42" y="26" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="57" y="42" text-anchor="middle" class="sm">2</text><rect x="240" y="26" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="259" y="42" text-anchor="middle" class="sm">3</text><rect x="74" y="26" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="89" y="42" text-anchor="middle" class="sm">3</text><rect x="280" y="26" width="38" height="24" class="soft-p" stroke="var(--line)"/><text x="299" y="42" text-anchor="middle" class="sm">6</text><rect x="106" y="26" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="121" y="42" text-anchor="middle" class="sm">1</text><rect x="320" y="26" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="339" y="42" text-anchor="middle" class="sm">7</text><rect x="10" y="52" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="25" y="68" text-anchor="middle" class="sm">3</text><rect x="200" y="52" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="219" y="68" text-anchor="middle" class="sm">4</text><rect x="42" y="52" width="30" height="24" class="soft-b" stroke="var(--line)"/><text x="57" y="68" text-anchor="middle" class="sm">4</text><rect x="240" y="52" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="259" y="68" text-anchor="middle" class="sm">10</text><rect x="74" y="52" width="30" height="24" class="soft-b" stroke="var(--line)"/><text x="89" y="68" text-anchor="middle" class="sm">1</text><rect x="280" y="52" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="299" y="68" text-anchor="middle" class="sm">14</text><rect x="106" y="52" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="121" y="68" text-anchor="middle" class="sm">2</text><rect x="320" y="52" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="339" y="68" text-anchor="middle" class="sm">17</text><rect x="10" y="78" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="25" y="94" text-anchor="middle" class="sm">2</text><rect x="200" y="78" width="38" height="24" class="soft-p" stroke="var(--line)"/><text x="219" y="94" text-anchor="middle" class="sm">6</text><rect x="42" y="78" width="30" height="24" class="soft-b" stroke="var(--line)"/><text x="57" y="94" text-anchor="middle" class="sm">1</text><rect x="240" y="78" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="259" y="94" text-anchor="middle" class="sm">13</text><rect x="74" y="78" width="30" height="24" class="soft-b" stroke="var(--line)"/><text x="89" y="94" text-anchor="middle" class="sm">5</text><rect x="280" y="78" width="38" height="24" class="soft-p" stroke="var(--line)"/><text x="299" y="94" text-anchor="middle" class="sm">22</text><rect x="106" y="78" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="121" y="94" text-anchor="middle" class="sm">3</text><rect x="320" y="78" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="339" y="94" text-anchor="middle" class="sm">28</text><rect x="10" y="104" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="25" y="120" text-anchor="middle" class="sm">4</text><rect x="200" y="104" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="219" y="120" text-anchor="middle" class="sm">10</text><rect x="42" y="104" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="57" y="120" text-anchor="middle" class="sm">2</text><rect x="240" y="104" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="259" y="120" text-anchor="middle" class="sm">19</text><rect x="74" y="104" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="89" y="120" text-anchor="middle" class="sm">1</text><rect x="280" y="104" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="299" y="120" text-anchor="middle" class="sm">29</text><rect x="106" y="104" width="30" height="24" class="soft-a" stroke="var(--line)"/><text x="121" y="120" text-anchor="middle" class="sm">6</text><rect x="320" y="104" width="38" height="24" class="soft-a" stroke="var(--line)"/><text x="339" y="120" text-anchor="middle" class="sm">41</text><text x="390" y="40" class="sm">Turuncu 2 × 2 kutunun toplamı:</text><text x="390" y="58" class="sm">A(2,2) − A(0,2) − A(2,0) + A(0,0)</text><text x="390" y="76" class="sm">= 22 − 6 − 6 + 1 = 11</text><text x="390" y="94" class="sm">(doğrudan: 4 + 1 + 1 + 5 = 11)</text><text x="10" y="148" class="sm">Kutu boyutu ne olursa olsun 4 arama ve 3 toplama/çıkarma: piksel başına O(1). Viola–Jones ve SURF bunu kullanır.</text></svg><figcaption>Şekil 2 — İntegral görüntüyle dikdörtgen toplamı (hesaplandı). Bir kez O(MN) ön hesap, sonra her kutu sorgusu sabit sürede.</figcaption></figure>

**El hesapları.** $I=\begin{bmatrix}1&2\\3&4\end{bmatrix}$ için $A(1,1)=1+2+3+4=\mathbf{10}$. 5 × 5'lik kutunun toplamı doğrudan 25 değerin toplanmasıyla $\mathbf{24}$ toplama ister; integral görüntüyle $\mathbf3$ toplama/çıkarma (dört arama). 1024 × 1024'lük 8 bitlik bir görüntünün integrali $255\cdot2^{20}$'ye kadar çıkar: taşmadan tutmak için en az $\mathbf{28}$ bit gerekir (OpenCV 32 bit tamsayı ya da float64 kullanır).

Aynı fikrin 1-B hâli **kayan toplamdır**: birikimli toplamın iki değerinin farkı (`cs[i + k] − cs[i]`), ya da pencere kayarken giren değeri ekleyip çıkanı çıkarmak (ek x3). Viola–Jones yüz bulucusu (M9.16) ve SURF (M9.14) yüzlerce farklı boyutta kutu toplamını bu sayede sabit sürede hesaplar.

### Donanımda süzgeç tasarımı {#donanim}

Ticari uygulamalarda uzamsal süzgeçler genellikle hızlandırıcılarda çalışır: yüksek düzeyde programlanabilir (GPU), düşük düzeyde programlanabilir (FPGA) ya da uygulamaya özgü (ASIC) yardımcı işlemciler. Özellikle FPGA ve ASIC'te şunlar tercih edilir:
1. **Ayrılabilir** süzgeçler (iyi): $m+n$ çarpma, $mn$ yerine.
2. **Tamsayı** katsayılar (daha iyi): kayan nokta birimi gerekmez.
3. **2'nin kuvveti** tamsayı katsayılar (en iyi): çarpma bir bit kaydırmaya iner; $[1,2,1]/4$ gibi binom çekirdekleri bu yüzden popülerdir.

Büyük görüntülerde (1080p ve üstü) büyük çekirdekler sorundur; asıl sorun çekirdeğin **yüksekliğidir**. Görüntü satır satır akarken $m$ satırlık bir çekirdek için $m-1$ satır **satır tamponunda** (line buffer, çip üstü bellek) tutulmalıdır: 1920 piksellik satırların her biri değerli yonga alanı harcar. Genişlik ise yalnız birkaç yazmaç gerektirir.

### PyTorch ve OpenCV'de {#pytorch}

| amaç | kod |
|---|---|
| korelasyon | `cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)` |
| konvolüsyon | `cv2.filter2D(img, -1, cv2.flip(k, -1), anchor=(kw - 1 - ax, kh - 1 - ay))` |
| PyTorch | `F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), w)` (korelasyon) |
| kutu | `cv2.boxFilter(img, -1, (k, k), normalize=True)`, `cv2.blur` |
| Gauss | `cv2.GaussianBlur(img, (k, k), sigma)`; çekirdek `cv2.getGaussianKernel(k, sigma)` |
| ayrılabilir | `cv2.sepFilter2D(img, -1, kx, ky)` |
| medyan | `cv2.medianBlur(img, k)` |
| integral | `cv2.integral(img)` → `(H + 1) × (W + 1)` |
| dolgu | `cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT_101)` |

### Özet {#ozet}

- Doğrusal süzgeç = çekirdekle çarpımlar toplamı; korelasyon ile konvolüsyon yalnız 180° döndürmeyle ayrılır (el hesabı 1 ve 9; Laplasyen 0). Dürtüye konvolüsyon çekirdeğin kopyasını verir.
- Kenar dolgusu: sıfır (koyu şerit), kopyalama, yansıtma (REFLECT_101 varsayılan), dairesel.
- Ayrılabilir = rank 1 = dış çarpım; kazanç $mn/(m+n)$ (3 × 3: 9 → 6; 11 × 11: 5.5, slayttaki 5.2 değil).
- Kutu ve Gauss (tek dairesel simetrik ve ayrılabilir çekirdek), normalizasyon, $\lceil6\sigma\rceil$ ve OpenCV σ kuralı (k = 7 → 1.4).
- Medyan ve ağırlıklı medyan (38 ve 42); tuz-biberde üstün, ince yapıyı siler; maks/min süzgeçleri.
- İntegral görüntü: kutu toplamı 4 aramayla (5 × 5: 24 → 3 işlem), 28 bit; kayan toplam.
- Donanım: ayrılabilirlik, tamsayı ve 2'nin kuvveti katsayılar; çekirdek yüksekliği satır tamponu ister.
