### Görüntü nedir? {#goruntu}

İnsanlar son derece görsel canlılardır: yazmadan, hatta belki konuşmadan çok önce şekiller çiziyorduk. Nesne tanıma, renk ayırt etme, değişikliği fark etme, bir sahneyi bölgelere ayırma ve hareketi izleme gibi üst düzey görsel işler insan beyni için zahmetsizdir. Görüntüler bilgiyi son derece verimli taşır; bu yüzden biriken insan bilgisinin büyük kısmı görseldir: tarihî ve teknik belgeler, sanat, eğlence, tıbbi kayıtlar, güvenlik kayıtları.

Bir **görüntü** iki boyutlu bir fonksiyondur, $f(x,y)$. $x$ ve $y$ uzamsal (düzlem) koordinatlardır; herhangi bir $(x,y)$ çiftindeki $f$ değerine o noktadaki **yoğunluk** ya da **gri seviye** denir. $x$, $y$ ve yoğunluk değerlerinin hepsi sonlu, ayrık büyüklüklerse görüntüye **dijital görüntü** denir. Dijital görüntü işleme, dijital görüntülerin bir bilgisayarla işlenmesidir. Dijital bir görüntü, her birinin belirli bir konumu ve değeri olan sonlu sayıda öğeden oluşur; bu öğelere resim öğesi, kısaca **piksel** denir.

**Görüntü işleme ve bilgisayarlı görü.** İki terim çoğu zaman birbirinin yerine kullanılır. Görüntü işleme için önerilmiş iki ayrım vardır: girdisi de çıktısı da görüntü olan işlemler; ya da çıktısı bir insanın gözle incelemesi için hazırlanan işlemler. Bilgisayarlı görünün nihai amacı ise insan görmesini taklit etmektir: görsel girdilerden öğrenmek, çıkarım yapmak ve eyleme geçmek.

**Görsel işlemenin düzeyleri.**
- **Alt düzey**: girdi de çıktı da görüntüdür (gürültü azaltma, karşıtlık iyileştirme, keskinleştirme).
- **Orta düzey**: girdi görüntü, çıktı ondan çıkarılan özniteliklerdir (kenarlar, konturlar, nesnelerin kimliği). Bölütleme, nesnelerin bilgisayarda işlenebilir bir biçimde betimlenmesi ve sınıflandırma buradadır.
- **Üst düzey**: insan görmesinin bilişsel işlevleri: tanınan nesnelerden anlam çıkarmak, görüntü altyazılama, anomali tespiti.

### Görüntüleme modaliteleri {#modalite}

Bir görüntü, düzenli bir 2-B uzamsal ızgarada (piksel konumlarında) örneklenmiş bir sinyaldir. Altta yatan sinyalin hangi özelliğinin yakalandığına ve nasıl bir algılayıcı kullanıldığına göre farklı **görüntüleme modaliteleri** vardır. Elektromanyetik spektrumda kısa dalga boyundan uzuna:
- **Gama ışını**: nükleer tıp ve astronomik gözlemler.
- **X-ışını**: tıbbi ve endüstriyel görüntüleme.
- **Morötesi**: litografi, endüstriyel denetim, mikroskopi, lazerler, biyolojik görüntüleme, astronomi.
- **Görünür bant**: ticari görüntüleme, cep telefonları, sanat, eğlence.
- **Kızılötesi**: gözetleme, gece görüşü (termal kameralar).
- **Mikrodalga**: görüntüleme radarları (sentetik açıklıklı radar).
- **Radyo**: manyetik rezonans görüntüleme (MRI).

Elektromanyetik olmayanlara bir örnek **ultrasondur**: mekanik (ses) dalgaları yakalanır.

:::uyarı Kaynak düzeltmesi: dalga boyları (slayt 19, 29)
Slayt 29 görünür bandı "yaklaşık 0.43 mm (mor) ile 0.79 mm (kırmızı)" arası verir; doğru birim **mikrometredir**: 0.43–0.79 µm, yani 430–790 nm (Gonzalez–Woods'un verdiği aralık). Slayt 19'da ultrason için "santimetre" yazıyor. Tıbbi ultrason 1–20 MHz civarında çalışır ve yumuşak dokuda ses hızı ≈ 1540 m/s'dir: dalga boyu $\lambda=c/f\approx0.08$–$1.5$ mm. Ultrasonun çözünürlüğü de bu milimetre altı dalga boyuyla sınırlıdır.
:::

Bir görüntü işleme sistemi algılayıcı, sayısallaştırıcı, depolama, bir işlemci (çoğu zaman GPU, FPGA ya da ASIC gibi hızlandırıcılar), yazılım ve görüntüleme birimlerinden oluşur.

### İnsan görme sistemi {#gorme}

Gözün en iç zarı **retinadır**; göz odaklandığında nesneden gelen ışık retinada görüntü oluşturur. Desen görmesini retina yüzeyine dağılmış ayrık ışık alıcıları sağlar. İki tür alıcı vardır:
- **Koniler**: her gözde 6–7 milyon kadardır, çoğu retinanın merkezindeki **fovea**dadır ve renge çok duyarlıdır. Her koni kendi sinir ucuna bağlı olduğu için ince ayrıntılar çözülebilir; kaslar gözü, ilgi bölgesinin görüntüsü foveaya düşene kadar döndürür. Koni görmesine **fotopik** (parlak ışık) görme denir.
- **Çubuklar**: sayıları çok daha fazladır (75–150 milyon) ve bütün retinaya yayılmıştır. Birkaç çubuk tek bir sinir ucuna bağlıdır; bu, ayrıntıyı azaltır ama genel bir görüntü verir. Çubuklar renk görmesine katılmaz ve düşük ışığa duyarlıdır: gündüz parlak renkli görünen nesneler ay ışığında renksiz biçimler olarak görünür. Buna **skotopik** (loş ışık) görme denir.

Üç tür koni vardır (kabaca kırmızı, yeşil, mavi). Göz en çok yeşile duyarlıdır.

:::uyarı Kaynak notu: "göz neden yeşile duyarlı?" (slayt 25)
Slayt bunu yeşil alıcıların bandının kırmızı ve mavi alıcılarınkiyle çok örtüşmesine bağlıyor. Daha doğru ifade şudur. Üç koni türünün tepeleri yaklaşık 420 nm (S), 530 nm (M) ve 560 nm'dedir (L, adı "kırmızı" olsa da tepesi sarı-yeşildedir). Gözün parlaklık duyarlılığı (fotopik ışık verimi eğrisi) ≈ 555 nm'de tepe yapar, çünkü M ve L konilerinin ikisi de en güçlü bu bölgede yanıt verir ve parlaklık algısına en çok katkıyı onlar yapar. Bayer filtresindeki yeşil fazlalığının gerekçesi de budur: parlaklık bilgisinin çoğu yeşil kanalda taşınır.
:::

Odaklama: sıradan bir kamerada merceğin odak uzaklığı sabittir; farklı uzaklıklara odaklanmak için mercek ile görüntü düzlemi (film ya da algılayıcı) arasındaki mesafe değiştirilir. İnsan gözünde tersi geçerlidir: merceğin merkezi ile retina arasındaki mesafe sabittir, gereken odak uzaklığı merceğin biçimi değiştirilerek sağlanır. Siliyer cisimdeki lifler uzak nesneler için merceği inceltir, yakın nesneler için kalınlaştırır.

### Işık, renk ve gri ton {#isik}

Işık, gözle algılanabilen bir elektromanyetik ışınımdır. Görünür bant yaklaşık 0.43–0.79 µm'dir ve kolaylık için altı geniş bölgeye ayrılır: mor, mavi, yeşil, sarı, turuncu, kırmızı. Hiçbir renk birden bitmez; her aralık bir sonrakine yumuşakça geçer.

Bir nesnede algılanan renk, nesnenin yansıttığı ışığın doğasıyla belirlenir. Görünür dalga boylarının hepsini dengeli yansıtan bir cisim beyaz görünür; sınırlı bir aralığı yeğleyen cisim bir renk tonunda görünür. Örneğin yeşil nesneler çoğunlukla 500–570 nm'deki ışığı yansıtır, diğer dalga boylarındaki enerjinin çoğunu soğurur.

Rengi olmayan ışığa **tek renkli** (akromatik) ışık denir; tek özniteliği yoğunluğudur. Tek renkli ışığın yoğunluğu siyahtan grilere ve beyaza değişir gibi algılandığı için **gri seviye** terimi yoğunlukla eş anlamlı kullanılır. Siyahtan beyaza değer aralığına **gri ölçek** denir ve tek renkli görüntülere **gri tonlu** görüntü denir.

### Görüntü edinimi ve oluşum modeli {#edinim}

Tek bir algılayıcı öğesine bir örnek **fotodiyottur**: silikondan yapılır ve çıkışı ışık yoğunluğuyla orantılı bir gerilimdir. Algılayıcının önüne bir filtre koymak seçiciliğini artırır: örneğin yeşil geçiren bir optik filtre yeşil bandı kayırır ve algılayıcının çıkışı yeşil ışık için daha güçlü olur.

Algılayıcı dizilerinde (CMOS) çoğu algılayıcı her pikselde bütün renk düzlemlerini örnekleyemez; her konumda yalnız bir renk bileşeni vardır. En yaygın desen **Bayer desenidir**: her 2 × 2 blokta iki yeşil, bir kırmızı, bir mavi. Yeşil kanal kırmızı ve maviye göre iki kat sık örneklenir, çünkü göz en çok yeşile duyarlıdır. Her pikselde bütün renk değerlerini elde etme işlemine **demozaikleme** denir (ek x4).

**Basit oluşum modeli.** Algılanan yoğunluk iki çarpanın ürünüdür:
$$f(x,y)=i(x,y)\,r(x,y),\qquad 0<i<\infty,\quad 0<r<1 .$$
$i$ sahneye düşen **aydınlatmadır** (ışık kaynağına bağlı), $r$ nesnelerin **yansıtıcılığıdır** (nesneye bağlı; 0 tam soğurma, 1 tam yansıma). Aydınlatma genellikle yavaş değişir, yansıtıcılık nesne sınırlarında keskin değişir. Bu ayrım, homomorfik süzgeçleme gibi yöntemlerin temelidir (M9.10).

### Örnekleme ve nicemleme {#ornekleme}

Algılayıcının çıkışı sürekli bir gerilim dalgasıdır. Dijital görüntü elde etmek için iki ayrıklaştırma gerekir:
- **Örnekleme**: koordinatların ayrıklaştırılması. Değerler bir ızgaranın noktalarında alınır, $M\times N$ piksel oluşur.
- **Nicemleme**: yoğunluk değerlerinin ayrıklaştırılması. Her değer $L=2^k$ seviyeden birine yuvarlanır.

<figure class="fig wide"><svg viewBox="0 0 640 206" role="img" aria-label="Sürekli bir tarama çizgisi eşit aralıklarla örneklenir ve her örnek 8 seviyeden en yakınına yuvarlanır"><path d="M50,175 L610,175" class="mute"/><path d="M50,15 L50,175" class="mute"/><path d="M50,175.0 L610,175.0" class="mute dash"/><path d="M50,152.1 L610,152.1" class="mute dash"/><path d="M50,129.3 L610,129.3" class="mute dash"/><path d="M50,106.4 L610,106.4" class="mute dash"/><path d="M50,83.6 L610,83.6" class="mute dash"/><path d="M50,60.7 L610,60.7" class="mute dash"/><path d="M50,37.9 L610,37.9" class="mute dash"/><path d="M50,15.0 L610,15.0" class="mute dash"/><path d="M50.0,95.0 L51.4,90.8 L52.8,86.7 L54.2,82.6 L55.6,78.6 L57.0,74.7 L58.4,71.0 L59.8,67.5 L61.2,64.2 L62.6,61.1 L64.0,58.2 L65.4,55.6 L66.8,53.3 L68.2,51.3 L69.6,49.5 L71.0,48.1 L72.4,46.9 L73.8,46.1 L75.2,45.5 L76.6,45.1 L78.0,45.0 L79.4,45.2 L80.8,45.5 L82.2,46.1 L83.6,46.8 L85.0,47.7 L86.4,48.6 L87.8,49.7 L89.2,50.8 L90.6,51.9 L92.0,53.1 L93.4,54.2 L94.8,55.2 L96.2,56.2 L97.6,57.0 L99.0,57.8 L100.4,58.4 L101.8,58.8 L103.2,59.1 L104.6,59.1 L106.0,59.0 L107.4,58.7 L108.8,58.2 L110.2,57.5 L111.6,56.7 L113.0,55.6 L114.4,54.4 L115.8,53.1 L117.2,51.6 L118.6,50.1 L120.0,48.5 L121.4,46.8 L122.8,45.1 L124.2,43.4 L125.6,41.7 L127.0,40.1 L128.4,38.6 L129.8,37.3 L131.2,36.1 L132.6,35.0 L134.0,34.2 L135.4,33.5 L136.8,33.2 L138.2,33.1 L139.6,33.2 L141.0,33.7 L142.4,34.5 L143.8,35.5 L145.2,36.9 L146.6,38.6 L148.0,40.6 L149.4,42.9 L150.8,45.5 L152.2,48.3 L153.6,51.4 L155.0,54.7 L156.4,58.2 L157.8,61.8 L159.2,65.7 L160.6,69.6 L162.0,73.6 L163.4,77.7 L164.8,81.8 L166.2,85.9 L167.6,89.9 L169.0,93.9 L170.4,97.8 L171.8,101.5 L173.2,105.1 L174.6,108.4 L176.0,111.6 L177.4,114.6 L178.8,117.3 L180.2,119.8 L181.6,122.0 L183.0,123.9 L184.4,125.6 L185.8,126.9 L187.2,128.1 L188.6,128.9 L190.0,129.6 L191.4,129.9 L192.8,130.1 L194.2,130.1 L195.6,129.9 L197.0,129.6 L198.4,129.1 L199.8,128.5 L201.2,127.9 L202.6,127.2 L204.0,126.6 L205.4,125.9 L206.8,125.3 L208.2,124.7 L209.6,124.3 L211.0,123.9 L212.4,123.7 L213.8,123.7 L215.2,123.8 L216.6,124.1 L218.0,124.6 L219.4,125.3 L220.8,126.2 L222.2,127.3 L223.6,128.5 L225.0,130.0 L226.4,131.6 L227.8,133.4 L229.2,135.4 L230.6,137.4 L232.0,139.6 L233.4,141.8 L234.8,144.1 L236.2,146.4 L237.6,148.7 L239.0,151.0 L240.4,153.2 L241.8,155.3 L243.2,157.3 L244.6,159.1 L246.0,160.7 L247.4,162.1 L248.8,163.3 L250.2,164.2 L251.6,164.9 L253.0,165.3 L254.4,165.3 L255.8,165.1 L257.2,164.5 L258.6,163.6 L260.0,162.4 L261.4,160.9 L262.8,159.1 L264.2,156.9 L265.6,154.5 L267.0,151.9 L268.4,149.0 L269.8,145.8 L271.2,142.5 L272.6,139.0 L274.0,135.4 L275.4,131.7 L276.8,127.9 L278.2,124.1 L279.6,120.2 L281.0,116.4 L282.4,112.7 L283.8,109.0 L285.2,105.4 L286.6,102.0 L288.0,98.8 L289.4,95.7 L290.8,92.8 L292.2,90.2 L293.6,87.7 L295.0,85.6 L296.4,83.6 L297.8,81.9 L299.2,80.5 L300.6,79.3 L302.0,78.4 L303.4,77.6 L304.8,77.1 L306.2,76.8 L307.6,76.7 L309.0,76.7 L310.4,76.8 L311.8,77.0 L313.2,77.3 L314.6,77.7 L316.0,78.1 L317.4,78.4 L318.8,78.8 L320.2,79.1 L321.6,79.3 L323.0,79.3 L324.4,79.3 L325.8,79.1 L327.2,78.7 L328.6,78.2 L330.0,77.4 L331.4,76.4 L332.8,75.3 L334.2,73.9 L335.6,72.3 L337.0,70.5 L338.4,68.5 L339.8,66.3 L341.2,64.0 L342.6,61.5 L344.0,58.8 L345.4,56.1 L346.8,53.3 L348.2,50.4 L349.6,47.5 L351.0,44.6 L352.4,41.7 L353.8,38.9 L355.2,36.2 L356.6,33.7 L358.0,31.2 L359.4,29.0 L360.8,27.0 L362.2,25.2 L363.6,23.6 L365.0,22.4 L366.4,21.4 L367.8,20.7 L369.2,20.4 L370.6,20.4 L372.0,20.7 L373.4,21.3 L374.8,22.3 L376.2,23.5 L377.6,25.1 L379.0,27.0 L380.4,29.1 L381.8,31.5 L383.2,34.1 L384.6,36.9 L386.0,39.9 L387.4,43.1 L388.8,46.3 L390.2,49.7 L391.6,53.1 L393.0,56.5 L394.4,60.0 L395.8,63.3 L397.2,66.6 L398.6,69.9 L400.0,73.0 L401.4,75.9 L402.8,78.7 L404.2,81.3 L405.6,83.7 L407.0,85.9 L408.4,87.9 L409.8,89.6 L411.2,91.1 L412.6,92.4 L414.0,93.5 L415.4,94.3 L416.8,95.0 L418.2,95.4 L419.6,95.7 L421.0,95.8 L422.4,95.8 L423.8,95.7 L425.2,95.5 L426.6,95.3 L428.0,95.0 L429.4,94.7 L430.8,94.5 L432.2,94.3 L433.6,94.2 L435.0,94.2 L436.4,94.3 L437.8,94.6 L439.2,95.0 L440.6,95.7 L442.0,96.5 L443.4,97.6 L444.8,98.9 L446.2,100.4 L447.6,102.1 L449.0,104.1 L450.4,106.3 L451.8,108.7 L453.2,111.3 L454.6,114.1 L456.0,117.0 L457.4,120.1 L458.8,123.4 L460.2,126.7 L461.6,130.0 L463.0,133.5 L464.4,136.9 L465.8,140.3 L467.2,143.7 L468.6,146.9 L470.0,150.1 L471.4,153.1 L472.8,155.9 L474.2,158.5 L475.6,160.9 L477.0,163.0 L478.4,164.9 L479.8,166.5 L481.2,167.7 L482.6,168.7 L484.0,169.3 L485.4,169.6 L486.8,169.6 L488.2,169.3 L489.6,168.6 L491.0,167.6 L492.4,166.4 L493.8,164.8 L495.2,163.0 L496.6,161.0 L498.0,158.8 L499.4,156.3 L500.8,153.8 L502.2,151.1 L503.6,148.3 L505.0,145.4 L506.4,142.5 L507.8,139.6 L509.2,136.7 L510.6,133.9 L512.0,131.2 L513.4,128.5 L514.8,126.0 L516.2,123.7 L517.6,121.5 L519.0,119.5 L520.4,117.7 L521.8,116.1 L523.2,114.7 L524.6,113.6 L526.0,112.6 L527.4,111.8 L528.8,111.3 L530.2,110.9 L531.6,110.7 L533.0,110.7 L534.4,110.7 L535.8,110.9 L537.2,111.2 L538.6,111.6 L540.0,111.9 L541.4,112.3 L542.8,112.7 L544.2,113.0 L545.6,113.2 L547.0,113.3 L548.4,113.3 L549.8,113.2 L551.2,112.9 L552.6,112.4 L554.0,111.6 L555.4,110.7 L556.8,109.5 L558.2,108.1 L559.6,106.4 L561.0,104.4 L562.4,102.3 L563.8,99.8 L565.2,97.2 L566.6,94.3 L568.0,91.2 L569.4,88.0 L570.8,84.6 L572.2,81.0 L573.6,77.3 L575.0,73.6 L576.4,69.8 L577.8,65.9 L579.2,62.1 L580.6,58.3 L582.0,54.6 L583.4,51.0 L584.8,47.5 L586.2,44.2 L587.6,41.0 L589.0,38.1 L590.4,35.5 L591.8,33.1 L593.2,30.9 L594.6,29.1 L596.0,27.6 L597.4,26.4 L598.8,25.5 L600.2,24.9 L601.6,24.7 L603.0,24.7 L604.4,25.1 L605.8,25.8 L607.2,26.7 L608.6,27.9 L610.0,29.3" class="cb w2"/><path d="M50.0,95.0 L50.0,83.6" class="ink"/><circle cx="50.0" cy="95.0" r="2.5" class="cb-f"/><rect x="46.0" y="79.6" width="8" height="8" class="ca-f"/><path d="M78.0,45.0 L78.0,37.9" class="ink"/><circle cx="78.0" cy="45.0" r="2.5" class="cb-f"/><rect x="74.0" y="33.9" width="8" height="8" class="ca-f"/><path d="M106.0,59.0 L106.0,60.7" class="ink"/><circle cx="106.0" cy="59.0" r="2.5" class="cb-f"/><rect x="102.0" y="56.7" width="8" height="8" class="ca-f"/><path d="M134.0,34.2 L134.0,37.9" class="ink"/><circle cx="134.0" cy="34.2" r="2.5" class="cb-f"/><rect x="130.0" y="33.9" width="8" height="8" class="ca-f"/><path d="M162.0,73.6 L162.0,83.6" class="ink"/><circle cx="162.0" cy="73.6" r="2.5" class="cb-f"/><rect x="158.0" y="79.6" width="8" height="8" class="ca-f"/><path d="M190.0,129.6 L190.0,129.3" class="ink"/><circle cx="190.0" cy="129.6" r="2.5" class="cb-f"/><rect x="186.0" y="125.3" width="8" height="8" class="ca-f"/><path d="M218.0,124.6 L218.0,129.3" class="ink"/><circle cx="218.0" cy="124.6" r="2.5" class="cb-f"/><rect x="214.0" y="125.3" width="8" height="8" class="ca-f"/><path d="M246.0,160.7 L246.0,152.1" class="ink"/><circle cx="246.0" cy="160.7" r="2.5" class="cb-f"/><rect x="242.0" y="148.1" width="8" height="8" class="ca-f"/><path d="M274.0,135.4 L274.0,129.3" class="ink"/><circle cx="274.0" cy="135.4" r="2.5" class="cb-f"/><rect x="270.0" y="125.3" width="8" height="8" class="ca-f"/><path d="M302.0,78.4 L302.0,83.6" class="ink"/><circle cx="302.0" cy="78.4" r="2.5" class="cb-f"/><rect x="298.0" y="79.6" width="8" height="8" class="ca-f"/><path d="M330.0,77.4 L330.0,83.6" class="ink"/><circle cx="330.0" cy="77.4" r="2.5" class="cb-f"/><rect x="326.0" y="79.6" width="8" height="8" class="ca-f"/><path d="M358.0,31.2 L358.0,37.9" class="ink"/><circle cx="358.0" cy="31.2" r="2.5" class="cb-f"/><rect x="354.0" y="33.9" width="8" height="8" class="ca-f"/><path d="M386.0,39.9 L386.0,37.9" class="ink"/><circle cx="386.0" cy="39.9" r="2.5" class="cb-f"/><rect x="382.0" y="33.9" width="8" height="8" class="ca-f"/><path d="M414.0,93.5 L414.0,83.6" class="ink"/><circle cx="414.0" cy="93.5" r="2.5" class="cb-f"/><rect x="410.0" y="79.6" width="8" height="8" class="ca-f"/><path d="M442.0,96.5 L442.0,106.4" class="ink"/><circle cx="442.0" cy="96.5" r="2.5" class="cb-f"/><rect x="438.0" y="102.4" width="8" height="8" class="ca-f"/><path d="M470.0,150.1 L470.0,152.1" class="ink"/><circle cx="470.0" cy="150.1" r="2.5" class="cb-f"/><rect x="466.0" y="148.1" width="8" height="8" class="ca-f"/><path d="M498.0,158.8 L498.0,152.1" class="ink"/><circle cx="498.0" cy="158.8" r="2.5" class="cb-f"/><rect x="494.0" y="148.1" width="8" height="8" class="ca-f"/><path d="M526.0,112.6 L526.0,106.4" class="ink"/><circle cx="526.0" cy="112.6" r="2.5" class="cb-f"/><rect x="522.0" y="102.4" width="8" height="8" class="ca-f"/><path d="M554.0,111.6 L554.0,106.4" class="ink"/><circle cx="554.0" cy="111.6" r="2.5" class="cb-f"/><rect x="550.0" y="102.4" width="8" height="8" class="ca-f"/><path d="M582.0,54.6 L582.0,60.7" class="ink"/><circle cx="582.0" cy="54.6" r="2.5" class="cb-f"/><rect x="578.0" y="56.7" width="8" height="8" class="ca-f"/><path d="M610.0,29.3 L610.0,37.9" class="ink"/><circle cx="610.0" cy="29.3" r="2.5" class="cb-f"/><rect x="606.0" y="33.9" width="8" height="8" class="ca-f"/><text x="44" y="179" text-anchor="end" class="sm">0</text><text x="44" y="19" text-anchor="end" class="sm">7</text><text x="50" y="196" class="sm">Turuncu: sürekli tarama çizgisi ve örnekleri. Mavi kareler: 3 bitlik nicemleme (8 seviye, kesikli çizgiler).</text></svg><figcaption>Şekil 1 — Örnekleme (uzay ekseninde ayrıklaştırma) ve nicemleme (değer ekseninde ayrıklaştırma). Örnek sayısı uzamsal çözünürlüğü, seviye sayısı (2ᵏ) bit derinliğini belirler.</figcaption></figure>

**Bit derinliği.** Her piksel konumundaki değeri temsil etmek için kullanılan bit sayısına **bit derinliği** denir. $2^k$ olası yoğunluk seviyesi olan bir görüntüye "$k$ bitlik görüntü" denir; 256 seviyeli bir görüntü 8 bitliktir. Piksel sayısına ve bit derinliğine göre sıkıştırılmamış bir görüntü megabaytlarca yer tutabilir. El hesabı: 1024 × 768, 8 bit gri görüntü $\mathbf{786\,432}$ bayt; 3 bitlik nicemleme $\mathbf8$ seviye; 200 değerini 3 bite indirmek $\lfloor200/32\rfloor\cdot32=\mathbf{192}$ (her seviye 32 gri değeri kapsar). Çok az bit **yanlış kontur** üretir: düzgün geçişler basamaklı görünür (ek x1).

**Uzamsal çözünürlük** sezgisel olarak görüntüde ayırt edilebilen en küçük ayrıntının ölçüsüdür. Piksel sayısına eşit değildir ve onunla ölçülemez: bulanık bir optik çok sayıda pikselle de ayrıntı çözemez. Gerçek bir tanım frekans uzayı kavramları ister (M9.10).

**Gösterim.** Bir $M\times N$ dijital görüntü bir matristir.

<figure class="fig wide"><svg viewBox="0 0 640 198" role="img" aria-label="İki koordinat kuralı: Gonzalez–Woods'ta x satır, y sütundur; OpenCV ve NumPy'da x sütun, y satırdır ve dizi img[y, x] diye indekslenir"><text x="30" y="18" class="tb">Gonzalez–Woods (slaytlar)</text><rect x="50" y="40" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="80" y="40" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="110" y="40" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="140" y="40" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="170" y="40" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="50" y="64" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="80" y="64" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="110" y="64" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="140" y="64" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="170" y="64" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="50" y="88" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="80" y="88" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="110" y="88" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="140" y="88" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="170" y="88" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="50" y="112" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="80" y="112" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="110" y="112" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="140" y="112" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="170" y="112" width="28" height="22" class="soft-b" stroke="var(--line)"/><rect x="140" y="64" width="28" height="22" class="cp-f" fill-opacity="0.5"/><text x="125" y="150" text-anchor="middle" class="sm">y (sütun) →</text><text x="220" y="90" class="sm">x (satır) ↓</text><text x="30" y="172" class="sm">f(x, y) = f(satır, sütun)</text><text x="30" y="188" class="sm">yeşil hücre: (x, y) = (1, 3)</text><text x="350" y="18" class="ta">OpenCV / NumPy (aşamalar)</text><rect x="370" y="40" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="400" y="40" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="430" y="40" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="460" y="40" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="490" y="40" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="370" y="64" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="400" y="64" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="430" y="64" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="460" y="64" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="490" y="64" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="370" y="88" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="400" y="88" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="430" y="88" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="460" y="88" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="490" y="88" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="370" y="112" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="400" y="112" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="430" y="112" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="460" y="112" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="490" y="112" width="28" height="22" class="soft-a" stroke="var(--line)"/><rect x="460" y="64" width="28" height="22" class="cp-f" fill-opacity="0.5"/><text x="445" y="150" text-anchor="middle" class="sm">x (sütun) →</text><text x="540" y="90" class="sm">y (satır) ↓</text><text x="350" y="172" class="sm">img[y, x] = img[satır, sütun]</text><text x="350" y="188" class="sm">yeşil hücre: (x, y) = (3, 1)</text></svg><figcaption>Şekil 3 — Aynı piksel iki kuralda farklı yazılır. Slaytlar Gonzalez–Woods kuralını, bu sitenin aşamaları OpenCV kuralını kullanır.</figcaption></figure>

:::uyarı Kaynak notu: iki koordinat kuralı
Slaytlar (Gonzalez–Woods) $x$'i aşağı doğru **satır**, $y$'yi sağa doğru **sütun** alır: $(x,y)=(\text{satır},\text{sütun})$, sağ elli koordinat sisteminin 90° döndürülmüş hâli. OpenCV, NumPy dizilerinin indeksleme düzeni ve bu sitenin aşamaları ise $x$'i **sütun**, $y$'yi **satır** alır ve diziyi `img[y, x]` diye indeksler. İki kural da "en hızlı değişen indeks en sağdakidir" bilgisini paylaşır (satır-ana bellek). Formülleri aktarırken hangi kuralın kullanıldığına dikkat etmek gerekir; el hesaplarında $(x, y)$ OpenCV anlamındadır.
:::

### İnterpolasyon {#interpolasyon}

İnterpolasyon eksik piksel değerlerini tahmin etme işlemidir. Büyütme, küçültme, döndürme ve geometrik düzeltme gibi işlerde kullanılır.
- **En yakın komşu**: en yakın pikselin değerini kopyala. Basamaklı (bloklu) görünür.
- **Doğrusal (1-B)**: iki en yakın komşudan geçen doğruyu kullan. **Bilineer (2-B)**: 1-B doğrusal interpolasyonu üç kez uygula: önce bir yönde (ör. $x$) iki kez, iki ara değer elde et; sonra diğer yönde bu ara değerlerle bir kez. Oranlar dört komşunun tek çarpanlarında birleştirilebilir.
- **Kübik (1-B)**: dört en yakın komşuya üçüncü derece bir polinom ($f(x)=ax^3+bx^2+cx+d$) uydur. Birinci türev koşulu, sol ve sağ bağlantılarda düzgünlüğü sağlar. **Bikübik (2-B)**: 1-B kübiği beş kez uygula: bir yönde dört kez (dört ara değer), diğer yönde bir kez.

<figure class="fig wide"><svg viewBox="0 0 640 236" role="img" aria-label="Bilineer interpolasyon: köşe değerleri 10, 20, 30, 40; x = 0.25'te satırlarda 12.5 ve 32.5, sonra y = 0.5'te 22.5"><rect x="120" y="30" width="160" height="160" class="mute"/><circle cx="120" cy="30" r="5" class="ca-f"/><text x="110" y="22" text-anchor="end" class="ta">10</text><circle cx="280" cy="30" r="5" class="ca-f"/><text x="290" y="22" text-anchor="start" class="ta">20</text><circle cx="120" cy="190" r="5" class="ca-f"/><text x="110" y="208" text-anchor="end" class="ta">30</text><circle cx="280" cy="190" r="5" class="ca-f"/><text x="290" y="208" text-anchor="start" class="ta">40</text><circle cx="160.0" cy="30" r="4" class="cp-f"/><circle cx="160.0" cy="190" r="4" class="cp-f"/><text x="160.0" y="20" text-anchor="middle" class="tp">12.5</text><text x="160.0" y="210" text-anchor="middle" class="tp">32.5</text><path d="M160.0,30 L160.0,190" class="cp dash"/><circle cx="160.0" cy="110.0" r="6" class="cb-f"/><text x="170.0" y="114.0" class="tb">22.5</text><text x="200.0" y="230" text-anchor="middle" class="sm">x →</text><text x="90" y="110.0" class="sm">y ↓</text><text x="330" y="60" class="sm">1. satırlarda x yönünde (a = 0.25):</text><text x="330" y="78" class="sm">   10 · 0.75 + 20 · 0.25 = 12.5</text><text x="330" y="96" class="sm">   30 · 0.75 + 40 · 0.25 = 32.5</text><text x="330" y="124" class="sm">2. y yönünde (b = 0.5):</text><text x="330" y="142" class="sm">   12.5 · 0.5 + 32.5 · 0.5 = 22.5</text><text x="330" y="170" class="sm">Tek formülde dört ağırlık:</text><text x="330" y="188" class="sm">(1−a)(1−b), a(1−b), (1−a)b, ab = 0.375, 0.125, 0.375, 0.125</text></svg><figcaption>Şekil 2 — Modülün el hesabı: bilineer interpolasyon üç tane 1-B doğrusal interpolasyondur (iki satırda, sonra sütunda).</figcaption></figure>

**El hesabı.** $I=\begin{bmatrix}10&20\\30&40\end{bmatrix}$ (satır $y$, sütun $x$). Bilineer $I(x=0.25,y=0.5)$: satırlarda $x$ yönünde $10\cdot0.75+20\cdot0.25=12.5$ ve $30\cdot0.75+40\cdot0.25=32.5$; sonra $y$ yönünde $0.5\cdot12.5+0.5\cdot32.5=\mathbf{22.5}$. En yakın komşu $I(x=0.6,y=0.4)$: en yakın piksel $(x,y)=(1,0)$, değer $\mathbf{20}$.

:::türetim İnterpolasyon bir çekirdekle evrişimdir
Ayrık örnekler $f_i$ ise interpolasyonlu değer $\hat f(x)=\sum_if_i\,K(x-i)$ biçimindedir; $K$ **interpolasyon çekirdeğidir**. En yakın komşu için $K$ genişliği 1 olan bir kutu, doğrusal için tabanı 2 olan bir üçgen. Keys'in (1981) kübik çekirdeği
$$K(t)=\begin{cases}(a+2)|t|^3-(a+3)|t|^2+1,&|t|\le1\\a|t|^3-5a|t|^2+8a|t|-4a,&1<|t|<2\\0,&\text{aksi}\end{cases}$$
$K(0)=1$ ve $K(\pm1)=K(\pm2)=0$'dır, yani örnek noktalarında değerleri aynen korur; birinci türevi süreklidir. $a=-0.5$ Taylor açılımında üçüncü dereceden doğruluk verir; OpenCV $a=-0.75$ kullanır (biraz daha keskin, biraz daha çok aşma). Kübik çekirdeğin negatif kanatları, keskin kenarların iki yanında hafif "halka" (ringing) üretir.
:::

**OpenCV'nin piksel merkezi kuralı.** Yeniden boyutlandırmada piksellerin değerleri kutuların **merkezlerinde** kabul edilir. Ölçek $s=$ kaynak/hedef boyu ise hedef $x$'in kaynak koordinatı $x_s=(x+0.5)s-0.5$'tir. El hesabı: 4 piksellik bir satır 2'ye küçülürken ($s=2$) hedef $x=0$'ın kaynağı $(0+0.5)\cdot2-0.5=\mathbf{0.5}$: iki kaynak pikselin tam ortası. Aşama b bu kuralı `cv2.resize` ile birebir yeniden üretiyor.

::lab l1

### Komşuluk ve mesafe {#komsuluk}

$(x,y)$'deki bir $p$ pikselinin iki yatay ve iki dikey komşusu, $(x\pm1,y)$ ve $(x,y\pm1)$, **4-komşulardır**: $N_4(p)$. Dört çapraz komşu $(x\pm1,y\pm1)$ $N_D(p)$ ile gösterilir; 4-komşularla birlikte **8-komşulardır**: $N_8(p)$. Bir noktanın komşularının konum kümesi onun komşuluğudur; $p$'yi içeriyorsa kapalı, içermiyorsa açıktır.

$D$'nin bir **metrik** olması için üç koşul gerekir: $D(p,q)\ge0$ (eşitlik yalnız $p=q$'da), $D(p,q)=D(q,p)$ ve üçgen eşitsizliği $D(p,s)\le D(p,q)+D(q,s)$. Üç yaygın metrik:
- **Öklid**: $D_e=\sqrt{(x-u)^2+(y-v)^2}$.
- **Şehir bloğu**: $D_4=|x-u|+|y-v|$; $D_4=1$ olan pikseller 4-komşulardır.
- **Satranç tahtası**: $D_8=\max(|x-u|,|y-v|)$; $D_8=1$ olanlar 8-komşulardır.

El hesabı: $p=(0,0)$, $q=(3,4)$: $D_e=\mathbf5$, $D_4=\mathbf7$, $D_8=\mathbf4$. Her zaman $D_8\le D_e\le D_4$'tür. Ek x2, bu mesafelerin bütün görüntü için iki geçişte hesaplandığı mesafe dönüşümünü yazıyor.

### Görüntüler üzerinde işlemler {#islemler}

**Eleman bazlı ve matris işlemleri.** Bir ya da daha çok görüntü üzerindeki eleman bazlı işlem piksel piksel yapılır (ör. iki görüntünün eleman bazlı çarpımı). Matris çarpımı ise ayrı bir işlemdir; görüntü işlemede aksi söylenmedikçe çarpım eleman bazlıdır. Görüntüler ayrıca küme işlemlerine (birleşim, kesişim, tümleyen) ve ikili görüntülerde mantıksal işlemlere (VE, VEYA, DEĞİL) girer.

**Uzamsal işlemler** doğrudan pikseller üzerinde yapılır ve üç türlüdür:
1. **Tek piksel işlemleri** (yaygın): her pikselin yoğunluğu bir dönüşüm fonksiyonuyla değiştirilir, $s=T(z)$. Gama düzeltmesi ve basit karşıtlık iyileştirme böyle yapılır. Pratik bir görüntü biçiminde olası yoğunluk sayısı sonlu olduğu için (8 bitte 256) bu işlem bir **arama tablosuyla** (LUT) uygulanabilir (M9.2).
2. **Komşuluk işlemleri** (yaygın): $S_{xy}$, $(x,y)$ merkezli bir komşuluğun koordinat kümesi olsun. Çıkıştaki piksel, girdinin $S_{xy}$'deki pikselleri üzerinde belirli bir işlemle hesaplanır. Örneğin $m\times n$'lik bir dikdörtgende ortalama: $g(x,y)=\frac1{mn}\sum_{(r,c)\in S_{xy}}f(r,c)$. Merkez pikselden piksele kaydırılır; doğrudan uygulama iç içe dört döngüdür (M9.3, keskinleştirme ve bulanıklaştırma).
3. **Geometrik dönüşümler** (daha seyrek): aşağıda.

### Geometrik dönüşümler {#geometri}

Geometrik dönüşümler piksellerin uzamsal düzenini değiştirir. Görüntüyü bir lastik tabakaya basıp tabakayı kurallara göre germek ya da daraltmak gibi düşünülebildiği için bunlara "lastik tabaka" dönüşümleri de denir. İki temel işlemden oluşur: (1) koordinatların uzamsal dönüşümü, (2) dönüştürülmüş piksellere yoğunluk atayan **interpolasyon**.

Koordinat dönüşümü $(x',y')=T(x,y)$ ile yazılır; örneğin $(x',y')=(x/2,y/2)$ görüntüyü iki yönde yarıya küçültür. İlgi alanımız **afin dönüşümlerdir** (6 parametre): ölçekleme, kaydırma, döndürme ve kesme (shear). Afin dönüşüm doğruları doğrulara götürür, paralellikleri ve bir doğru üzerindeki uzunluk oranlarını korur. Kaydırma dışındakiler bir $2\times2$ matrisle yazılır; kaydırma bir sabit vektör eklemeyi gerektirir. **Homojen koordinatlarla** dördü tek bir $3\times3$ matriste toplanır:
$$\begin{bmatrix}x'\\y'\\1\end{bmatrix}=\begin{bmatrix}a_{11}&a_{12}&a_{13}\\a_{21}&a_{22}&a_{23}\\0&0&1\end{bmatrix}\begin{bmatrix}x\\y\\1\end{bmatrix}.$$
Birden çok dönüşüm, matrislerin çarpımıyla tek bir matriste birleştirilir.

**İleri ve ters eşleme.** Dönüşümü uygulamanın iki yolu vardır. **İleri eşleme** girdinin piksellerini tarar ve her $(x,y)$ için çıkıştaki $(x',y')$ konumunu hesaplar. Ama bu işe yaramaz: (1) iki ya da daha çok girdi pikseli aynı çıkış konumuna düşebilir (değerler nasıl birleştirilecek?); (2) bazı çıkış konumlarına hiç piksel düşmeyebilir (delik). **Ters eşleme** çıkış piksellerini tarar ve her $(x',y')$ için girdideki konumu $(x,y)=A^{-1}(x',y')$ ile hesaplar, sonra en yakın girdi pikselleri arasında interpolasyon yapar. Ters eşleme uygulamada tek doğru yoldur ve ticari gerçekleştirimler (MATLAB, OpenCV'nin `warpAffine`'i) böyle çalışır.

Aşağıdaki deneyde yalnız 45° döndürmede bile ileri eşleme 48 × 48'lik bir görüntüde 321 delik ve 305 çakışma bırakıyor; ters eşleme hiç bırakmıyor.

::lab l2

### Görüntü kayıtlama {#kayitlama}

**Görüntü kayıtlama** (registration), aynı sahnenin iki ya da daha çok görüntüsünü hizalar. Bir girdi ve bir başvuru görüntüsü vardır; amaç girdiyi geometrik olarak dönüştürüp başvuruyla hizalı bir çıkış üretmektir. Önceki bölümün aksine dönüşüm bilinmez ve **kestirilmelidir**. Örnekler: farklı sistemlerle aynı anda alınmış görüntüler (MRI ve PET), farklı konumlardaki kameralar ya da aynı aletle farklı zamanlarda alınmış uydu görüntüleri. Bakış açısı, uzaklık, yönelim, algılayıcı çözünürlüğü ve nesne konumundaki kaymalardan doğan geometrik bozulmalar düzeltilmelidir. Yakından ilişkili bir problem **görüntü dikmedir** (panorama, M9.15).

**Bağlantı noktaları.** Ana yaklaşımlardan biri **bağlantı noktalarıdır** (tie, key ya da control points): konumları hem girdide hem başvuruda kesin bilinen karşılık noktaları. Etkileşimli seçilebilir ya da otomatik bulunabilir. Bazı sistemler algılayıcıya gömülü küçük metal parçalar gibi fiziksel işaretler (réseau ya da fiducial işaretleri) taşır.

**Dönüşümü kestirmek bir modelleme problemidir.** Dört bağlantı noktası çiftiyle basit bir **bilineer** model:
$$x=c_1v+c_2w+c_3vw+c_4,\qquad y=c_5v+c_6w+c_7vw+c_8 .$$
Dört nokta çifti sekiz denklem verir ve sekiz katsayı çözülür. Uygulamada bu model de **ters yönde** kurulur: çıkıştaki her $(x,y)$ için girdideki $(v,w)$'yi veren katsayılar ($c'_1,\dots,c'_8$) kestirilir ve hesaplanan ızgara dışı konumlarda interpolasyon yapılır. Dört nokta yetmezse çok sayıda nokta seçilir ve her dört noktanın oluşturduğu dörtgen ayrı bir alt görüntü gibi dönüştürülür; daha karmaşık modeller (en küçük kareler polinomları) de kullanılabilir. Afin model için en az üç nokta çifti gerekir (6 bilinmeyen; ek x3).

**Gerçekçi bir kayıtlama akışı:**
1. Görüntülerde anahtar noktaları bul (birden çok yönde yüksek gradyanlı pikseller, ör. köşeler; Harris ve Stephens 1988, M9.7).
2. Bu noktalar etrafındaki yamalardan betimleyiciler çıkar (yamanın piksel değerleri ya da daha iyisi gradyan histogramları; SIFT 1999, SURF; M9.14).
3. Betimleyicilerle anahtar noktaları eşleştir.
4. Eşleşmelerden dönüşümün denklemlerini kur ve parametreleri çöz (aykırı eşleşmelere karşı RANSAC, M9.15).
5. Girdiyi bu parametrelerle başvuruya eşle.

Aşama d bunu farklı bir yoldan, **doğrudan** yapıyor: dönüşüm parametrelerini, `grid_sample` ile türevlenebilir hâle getirilen ters eşlemenin piksel MSE'sini gradyan inişiyle en küçükleyerek buluyor (M6'daki autograd'ın görüntü işlemede kullanımı).

### Vektör ve matris gösterimi {#vektor}

Bir RGB renkli görüntünün her pikseli üç bileşenlidir ve bir sütun vektörü olarak yazılabilir: $\mathbf z=(z_1,z_2,z_3)^\top$ (kırmızı, yeşil, mavi). $M\times N$ bir RGB görüntü üç bileşen görüntüsüdür ya da $MN$ tane $3\times1$ vektördür. $n$ bantlı çok bantlı (multispectral) ve hiperspektral görüntüler $n$ boyutlu vektörler verir (M9.9).

Görüntünün tamamı da bir vektör olarak ele alınabilir: $M\times N$ görüntü, sütunları alt alta dizilerek $MN\times1$'lik bir vektör olur. O zaman görüntüye uygulanan geniş bir doğrusal işlemler sınıfı
$$\mathbf g=\mathbf H\mathbf f+\mathbf n$$
ile yazılır: $\mathbf f$ girdi, $\mathbf n$ gürültü, $\mathbf g$ işlenmiş görüntü, $\mathbf H$ doğrusal işlemi temsil eden $MN\times MN$'lik matris. Restorasyon (M9.5) bu denklemi $\mathbf f$ için çözmektir.

### Dönüşüm uzayında işleme {#donusum}

Şimdiye kadarki yaklaşımlar doğrudan uzamsal alanda çalışır. Bazı görevler en iyi şöyle yapılır: girdiyi bir dönüşüm uzayına taşı, görevi orada yap, ters dönüşümle uzamsal alana dön. Genel bir 2-B doğrusal dönüşüm
$$T(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)\,r(x,y,u,v),\qquad f(x,y)=\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}T(u,v)\,s(x,y,u,v)$$
biçimindedir; $r$ ileri, $s$ ters dönüşüm çekirdeğidir, ikisi birlikte bir **dönüşüm çiftidir**. En önemlisi **Fourier dönüşümüdür**: $r=e^{-j2\pi(ux/M+vy/N)}$, $s=\frac1{MN}e^{j2\pi(ux/M+vy/N)}$. Bir çekirdek $r(x,y,u,v)=r_1(x,u)r_2(y,v)$ ise **ayrılabilir**, $r_1=r_2$ ise **simetriktir**. Fourier çekirdekleri ikisini de sağlar; bu, 2-B dönüşümün satırlarda ve sütunlarda 1-B dönüşümlerle hesaplanmasını sağlar.

:::türetim Neden frekans uzayı? Özvektörlerden özfonksiyonlara
Bir matrisin özvektörleri biliniyorsa her vektör onlarda ayrıştırılabilir ve matrisin etkisi basitleşir: $\mathbf x=\sum k_i\mathbf e_i\Rightarrow\mathbf A\mathbf x=\sum k_i\lambda_i\mathbf e_i$. **Doğrusal ve uzayda değişmez** (LSI; zamanda LTI) bir sistem, ölçekleme ve süperpozisyonu sağlar ve kaydırmayla değişmez: $T[x(s-q)]=y(s-q)$. Böyle bir sistem dürtü yanıtı $h$ ile evrişimle tanımlanır: $y[n]=\sum_kx[k]h[n-k]$. Evrişimi hesaplamak ve yorumlamak, dolayısıyla $h$'yi doğrudan tasarlamak zordur. Ama karmaşık üsteller LSI sistemlerin **özfonksiyonlarıdır**: $x[n]=e^{j\omega n}$ için
$$y[n]=\sum_kh[k]e^{j\omega(n-k)}=\underbrace{\Big(\sum_kh[k]e^{-j\omega k}\Big)}_{H(\omega)}e^{j\omega n}.$$
Girdi karmaşık üstellerin toplamı olarak yazılırsa sistemin etkisi her frekansta bir çarpmaya iner: **uzayda evrişim, frekansta çarpımdır**. Fourier dönüşümü, sinyalin bu özfonksiyonlara izdüşümüdür; bir süzgeç, istenen frekansları güçlendiren ya da bastıran çarpanlar olarak tasarlanır (M9.10).
:::

:::uyarı Kaynak notu: karmaşık iç çarpım (slayt 110)
Slayt iç çarpımı $\sum_nf[n]g[n]$ olarak verir; bu yalnız gerçel fonksiyonlar için doğrudur. Karmaşık fonksiyonlarda iç çarpım $\sum_nf[n]\,\overline{g[n]}$'dir (ikinci fonksiyonun eşleniği). Fourier dönüşümünün çekirdeğinde $e^{-j\ldots}$ bulunmasının nedeni budur: $f$'nin $e^{+j\omega n}$ taban fonksiyonuna izdüşümü $\sum f[n]e^{-j\omega n}$'dir. Slaytın "Fourier analizinin tek dezavantajı zamandaki çözünürlüğün kaybıdır" notu da tamamlanmalı: kısa zamanlı Fourier dönüşümü ve dalgacıklar bu ödünleşimi ayarlar.
:::

### Pikseller rastgele değişkenler olarak {#olasilik}

Görüntü yoğunlukları rastgele büyüklükler gibi ele alınabilir. $z_k$ ($k=0,\dots,L-1$) olası yoğunluklar, $n_k$ $z_k$'nin görüntüde kaç kez geçtiği olsun. $M\times N$'lik görüntüde $z_k$'nin olasılığı $p(z_k)=n_k/(MN)$ ile kestirilir (normalize **histogram**, M9.2). Buradan ortalama ve varyans:
$$m=\sum_{k=0}^{L-1}z_kp(z_k),\qquad\sigma^2=\sum_{k=0}^{L-1}(z_k-m)^2p(z_k).$$
Ortalama genel parlaklığın, varyans (ya da standart sapma) karşıtlığın ölçüsüdür.

### PyTorch ve OpenCV'de {#pytorch}

| amaç | kod |
|---|---|
| yeniden boyutlandırma | `cv2.resize(img, (W2, H2), interpolation=cv2.INTER_LINEAR)` (boyut sırası (genişlik, yükseklik)) |
| PyTorch'ta | `F.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)` (OpenCV ile aynı merkez kuralı) |
| afin dönüşüm | `cv2.warpAffine(img, M, (W, H))` (M: 2 × 3, ters eşleme içeride) |
| türevlenebilir eşleme | `grid = F.affine_grid(theta, size, align_corners=False); F.grid_sample(x, grid)` |
| dönüşüm kestirimi | `cv2.estimateAffine2D(src_pts, dst_pts)`, `cv2.getPerspectiveTransform` |
| histogram | `np.bincount(img.ravel(), minlength=256) / img.size` |

### Özet {#ozet}

- Dijital görüntü: örneklenmiş ve nicemlenmiş $f(x,y)$; $f=i\cdot r$ (aydınlatma × yansıtıcılık). Bit derinliği $k$ → $2^k$ seviye (1024 × 768 × 8 bit = 786 432 bayt; 200 → 192).
- Modaliteler gama ışınından radyoya; görünür bant 0.43–0.79 µm (slayttaki "mm" düzeltildi); ultrasonun dalga boyu milimetre altı.
- Göz: koniler (fovea, renk, fotopik), çubuklar (loş ışık, skotopik); Bayer deseninde iki kat yeşil.
- İnterpolasyon = çekirdekle evrişim: en yakın (kutu), bilineer (üçgen, el hesabı 22.5), bikübik (Keys, $a=-0.75$ OpenCV); OpenCV kuralı $x_s=(x+0.5)s-0.5$.
- Komşuluk $N_4$, $N_D$, $N_8$; metrikler $D_e$, $D_4$, $D_8$ (5, 7, 4).
- Afin dönüşüm homojen $3\times3$ matris; ileri eşleme delik ve çakışma bırakır, ters eşleme kullanılır.
- Kayıtlama: bağlantı noktaları, bilineer model (8 katsayı), gerçekçi akış (anahtar nokta, betimleyici, eşleme, kestirim).
- Dönüşüm uzayı: Fourier, ayrılabilir çekirdekler; karmaşık üsteller LSI sistemlerin özfonksiyonlarıdır, evrişim çarpıma döner.
- Histogram olasılık olarak; ortalama ve varyans.
