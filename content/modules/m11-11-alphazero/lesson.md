### Go ve neden zordu {#go}

Go iki oyunculu bir tahta oyunudur. Basitleştirilmiş kuralları şöyledir:

- Siyah ve beyaz sırayla boş bir kesişim noktasına kendi renklerinden bir taş koyar.
- Birbirine bağlı taşlardan oluşan bir grubun hiç **özgürlüğü** (komşu boş kesişimi) kalmazsa grup ele geçirilir ve tahtadan kaldırılır.
- Oyun sonunda en çok alanı denetleyen oyuncu kazanır.

AlphaGo'nun maçlarında Çin kuralları (alan sayımı) ve 7.5 puanlık **komi** kullanıldı. Komi, ikinci oynayan beyaza verilen ek puandır. Kaynaktaki "en çok kesişimi denetleyen kazanır" ifadesi bu ayrıntıyı atlıyor. Komi olmadan ilk oynayanın avantajı belirleyicidir.

Go, oyun yapay zekâsının en zor hedeflerinden biri sayılıyordu. İki nedeni vardır:

- **Arama ağacı devasadır.** Dallanma çarpanı $b\approx250$, oyun uzunluğu $d\approx150$'dir. Tam ağaç $b^d\approx10^{360}$ yaprak demektir. Tahta $3^{361}\approx10^{172}$ dizilime izin verir (M11.6).
- **Konumu değerlendirmek zordur.** Satrançta taş değerlerinden kaba ama işe yarar bir değerlendirme fonksiyonu kurulabilir. Go'da bir konumun kimin lehine olduğunu söylemek ustalar için bile sezgiye dayanır.

Bu iki sorun M11.10'daki üç fikrin karşılığıdır. **Genişliği** daraltmak için umut veren hamlelere odaklanılır; bunu bir **politika** sağlar. **Derinliği** kesmek için yaprakta konum değerlendirilir; bunu bir **değer** fonksiyonu sağlar. AlphaGo ikisini de derin ağlarla öğrendi.

DeepMind 2010'da Londra'da Demis Hassabis, Shane Legg ve Mustafa Suleyman tarafından kuruldu ve 2014'te Google tarafından satın alındı. AlphaGo'nun sürümleri şöyledir:

- **AlphaGo Fan:** Ekim 2015'te Avrupa şampiyonu Fan Hui'yi 5–0 yendi. Bilgisayarın tam tahtada, handikapsız bir profesyoneli ilk yenişiydi.
- **AlphaGo Lee:** Mart 2016'da 9. dan Lee Sedol'u 4–1 yendi.
- **AlphaGo Master:** Mayıs 2017'de dünya bir numarası Ke Jie'yi 3–0 yendi.
- **AlphaGo Zero:** hiç insan verisi kullanmadan bütün öncüllerini geçti.

### AlphaGo: dört aşama {#alphago}

AlphaGo (Silver ve ark. 2016) dört bileşeni sırayla eğitir ve aramada birleştirir (Şekil 1).

<figure class="fig wide"><svg viewBox="0 0 640 246" role="img" aria-label="AlphaGo ve AlphaGo Zero eğitim boru hatlarının şeması"><defs><marker id="ah11" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs><text x="14" y="16" class="sm">AlphaGo (2016)</text><rect x="14" y="24" width="140" height="66" rx="6" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="22" y="40" class="sm">30 milyon KGS konumu</text><text x="22" y="55" class="sm">→ SL politika p<tspan baseline-shift="-3" font-size="75%">σ</tspan></text><text x="22" y="70" class="sm">13 katman, %57.0 isabet</text><text x="22" y="85" class="sm">50 GPU, 3 hafta</text><rect x="174" y="24" width="140" height="66" rx="6" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="182" y="40" class="sm">kendine karşı oyun</text><text x="182" y="55" class="sm">→ RL politika p<tspan baseline-shift="-3" font-size="75%">ρ</tspan></text><text x="182" y="70" class="sm">REINFORCE, z = ±1</text><text x="182" y="85" class="sm">SL'yi %80 yener</text><rect x="334" y="24" width="140" height="66" rx="6" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="342" y="40" class="sm">30 milyon konum, her</text><text x="342" y="55" class="sm">biri ayrı oyundan</text><text x="342" y="70" class="sm">→ değer ağı v<tspan baseline-shift="-3" font-size="75%">θ</tspan></text><text x="342" y="85" class="sm">test MSE 0.234</text><rect x="494" y="24" width="132" height="66" rx="6" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="502" y="40" class="sm">APV-MCTS (PUCT)</text><text x="502" y="55" class="sm">önsel: p<tspan baseline-shift="-3" font-size="75%">σ</tspan></text><text x="502" y="70" class="sm">yaprak: (1 − λ)v<tspan baseline-shift="-3" font-size="75%">θ</tspan> + λz</text><text x="502" y="85" class="sm">λ = 0.5, rollout p<tspan baseline-shift="-3" font-size="75%">π</tspan></text><path d="M154,57 L172,57" class="mute" stroke="currentColor" marker-end="url(#ah11)"/><path d="M314,57 L332,57" class="mute" stroke="currentColor" marker-end="url(#ah11)"/><path d="M474,57 L492,57" class="mute" stroke="currentColor" marker-end="url(#ah11)"/><text x="14" y="118" class="sm">AlphaGo Zero (2017): insan verisi yok, rollout yok, tek ağ</text><rect x="14" y="126" width="190" height="80" rx="6" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="22" y="142" class="sm">kendine karşı oyun α<tspan baseline-shift="-3" font-size="75%">θ</tspan>*</text><text x="22" y="157" class="sm">1600 simülasyon (≈ 0.4 s)</text><text x="22" y="172" class="sm">ilk 30 hamle τ = 1, sonra τ → 0</text><text x="22" y="187" class="sm">kökte Dir(0.03), ε = 0.25</text><rect x="224" y="126" width="190" height="80" rx="6" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="232" y="142" class="sm">eğitim: (s, π, z) üçlüleri</text><text x="232" y="157" class="sm">l = (z − v)² − πᵀlog p + c‖θ‖²</text><text x="232" y="172" class="sm">2048'lik yığınlar, c = 10⁻⁴</text><text x="232" y="187" class="sm">1000 adımda bir yeni ağ</text><rect x="434" y="126" width="192" height="80" rx="6" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="442" y="142" class="sm">değerlendirici: 400 oyun</text><text x="442" y="157" class="sm">yeni ağ %55'ten çok</text><text x="442" y="172" class="sm">kazanırsa en iyi ağ olur</text><text x="442" y="187" class="sm">(AlphaZero'da bu adım yok)</text><path d="M204,166 L222,166" class="mute" stroke="currentColor" marker-end="url(#ah11)"/><path d="M414,166 L432,166" class="mute" stroke="currentColor" marker-end="url(#ah11)"/><path d="M530,206 L530,222 L110,222 L110,208" class="mute" stroke="currentColor" fill="none" marker-end="url(#ah11)"/><text x="250" y="236" class="sm">en iyi ağ yeni oyunları üretir: politika yinelemesi döngüsü</text></svg><figcaption>Şekil 1 — İki sistemin boru hattı. AlphaGo dört ayrı bileşeni sırayla eğitir ve aramada birleştirir. AlphaGo Zero tek bir ağ ile arama ve öğrenmeyi kapalı bir döngüye bağlar. Sayılar makalelerden (Silver ve ark. 2016, 2017).</figcaption></figure>

**1. Denetimli politika ağı $p_\sigma$.** Uzman hamlelerini taklit eder (M11.9'daki en çok olabilirlik):
$$\Delta\sigma\propto\frac{\partial\log p_\sigma(a\mid s)}{\partial\sigma}.$$
Ayrıntılar şöyledir:

- Veri, KGS Go sunucusundaki 6–9. dan oyunculardan 160 000 oyunluk 29.4 milyon konumdur.
- Girdi $19\times19\times48$'lik bir öznitelik yığınıdır: taşın rengi, özgürlükler, son hamleler ve benzerleri.
- Ağ 13 katmanlı bir evrişimli ağdır ve çıktısı bütün kesişimler üzerinde bir softmax'tır.
- İsabet bütün özniteliklerle %57.0, yalnız ham tahtayla %55.7'dir; başka grupların o tarihteki en iyi sonucu %44.4'tü.
- Eğitim 50 GPU'da 3 hafta sürdü (340 milyon adım).

Ayrıca hızlı bir **rollout politikası** $p_\pi$ eğitildi. Elle tasarlanmış örüntü özniteliklerinin doğrusal softmax'ıdır. İsabeti yalnız %24.2'dir, ama bir hamleyi 3 ms yerine 2 µs'de seçer.

**2. Pekiştirmeli politika ağı $p_\rho$.** $p_\sigma$ ile başlatılır ve kendine karşı oyunla REINFORCE kullanılarak geliştirilir. Ödül yalnız oyun sonundadır: $z_t=+1$ (kazanç) ya da $-1$ (kayıp), $\gamma=1$. Güncelleme
$$\Delta\rho\propto\frac{\partial\log p_\rho(a_t\mid s_t)}{\partial\rho}z_t$$
olur. Burada $a_t$, **kendine karşı oyunda gerçekten oynanan** hamledir. Rakip, politikanın eski sürümlerinden oluşan bir havuzdan rastgele seçilir (her 500 yinelemede havuza yeni bir sürüm eklenir). Böylece politika tek bir rakibe aşırı uyum sağlamaz. Eğitim 128 oyunluk 10 000 mini yığında, 50 GPU ile bir gün sürdü. $p_\rho$, $p_\sigma$'yı oyunların %80'inden fazlasında yener. Hiç arama yapmadan güçlü açık kaynaklı program Pachi'yi %85 yener. Önceki en iyi evrişimli ağ Pachi'ye karşı ancak %11 kazanıyordu.

**3. Değer ağı $v_\theta$.** $v_\theta(s)\approx v^{p_\rho}(s)$, yani $s$'den iki tarafın da $p_\rho$ ile oynadığı durumdaki beklenen sonuçtur. Kare hatayla regresyon yapılır:
$$\Delta\theta\propto\frac{\partial v_\theta(s)}{\partial\theta}\bigl(z-v_\theta(s)\bigr).$$
Bu, $(z-v_\theta(s))^2$'nin gradyan **inişidir**. İlk denemede KGS'deki bütün oyunların bütün konumlarıyla eğitildi. Ağ aşırı uydu: eğitim MSE'si 0.19, test MSE'si 0.37 çıktı. Aynı oyunun ardışık konumları neredeyse aynıdır ve aynı sonucu paylaşır; ağ konumu değerlendirmek yerine oyunu ezberler. Çözüm, her biri **ayrı bir kendine karşı oyundan** örneklenmiş 30 milyon konumdur. Bu veriyle MSE eğitimde 0.226, testte 0.234 oldu. Tek bir değer ağı değerlendirmesi, $p_\rho$ ile yapılan Monte Carlo rollout'larının doğruluğuna yaklaşır ve 15 000 kat daha az hesap ister.

:::uyarı Kaynak düzeltmesi (lect_8): AlphaGo'nun RL ve değer ağı güncellemeleri
Slaytlardaki iki formül hatalıdır:

1. **RL politika gradyanı.** Slayt, gradyanı $\partial\sum_i\log\pi_\theta(a_i^*\mid s_i)/\partial\theta\cdot\gamma^nG_n$ diye yazıyor. $a_i^*$ uzman (en iyi) hamlesidir ve denetimli adıma aittir. Kendine karşı oyunda uzman yoktur. Gradyan, **oynanan** hamlenin log-olabilirliğinin sonuçla çarpımıdır: $\nabla\log p_\rho(a_t\mid s_t)z_t$.
2. **Değer ağı.** Slayt amacı $(V_w(s)-G)^2$'yi en küçüklemek olarak tanımlıyor. Gradyanı $\nabla_w=\frac{\partial V_w}{\partial w}(V_w-G)$ diye doğru yazıyor, ama güncellemeyi $w\leftarrow w+\alpha\nabla_w$ yapıyor. Bu, kare hatayı **artırır**. Doğrusu $w\leftarrow w-\alpha\nabla_w$, ya da eşdeğeri $w\leftarrow w+\alpha\frac{\partial V_w}{\partial w}(G-V_w)$'dir.

Ayrıca slayttaki Ke Jie zaferi (Mayıs 2017) AlphaGo'nun "Master" sürümüne aittir; Lee Sedol'u yenen sürümden farklıdır.
:::

**4. Arama.** AlphaGo, ağları bir MCTS türünde (APV-MCTS) birleştirir. Ağacın her kenarı $(s,a)$ için üç nicelik tutulur: eylem değeri $Q(s,a)$, ziyaret sayısı $N(s,a)$ ve önsel olasılık $P(s,a)$. Her simülasyon dört evreden geçer:

- **Seçim:** $a_t=\arg\max_a\bigl(Q(s_t,a)+u(s_t,a)\bigr)$ ile inilir. Bonus PUCT kuralıdır:
  $$u(s,a)=c_{\text{puct}}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}.$$
  Başta önseli yüksek ve az denenmiş hamleler seçilir; asimptotik olarak değeri yüksek hamleler seçilir.
- **Genişletme:** bir kenarın ziyaret sayısı bir eşiği aşınca ardıl konum ağaca eklenir. Önselleri $P(s',a)=p_\sigma(a\mid s')$ ile doldurulur.
- **Değerlendirme:** yaprak iki yolla değerlendirilir ve karıştırılır:
  $$V(s_L)=(1-\lambda)v_\theta(s_L)+\lambda z_L.$$
  $z_L$, yapraktan $p_\pi$ ile oynanan hızlı bir oyunun sonucudur.
- **Geri yayılım:** $N$ ve $Q$, yoldaki bütün kenarlarda güncellenir; $Q$, alt ağaçtaki değerlendirmelerin ortalamasıdır.

Arama bitince kökte en çok ziyaret edilen hamle oynanır. Örnek: $\lambda=0.5$, $v_\theta=0.2$, $z=1$ ise yaprak değeri $0.6$'dır. Paralel aramada aynı yolu birden çok iş parçacığının izlemesini önlemek için **sanal kayıp** kullanılır.

İki sonuç dikkat çekicidir:

- **Önsel olarak SL ağı daha iyidir.** Önselde daha güçlü $p_\rho$ yerine $p_\sigma$ kullanmak daha iyi sonuç verdi. Makalenin açıklaması şudur: insanlar umut veren hamlelerden oluşan **çeşitli** bir küme seçer, RL ise tek bir en iyi hamleye yoğunlaşır. Arama için çeşitlilik daha değerlidir. Değer ağında ise $p_\rho$'dan türetilen değer, $p_\sigma$'dan türetilenden daha iyiydi.
- **Karışım her iki bileşenden de iyidir.** $\lambda=0.5$, yalnız değer ağına ($\lambda=0$) ya da yalnız rollout'a ($\lambda=1$) karşı oyunların %95'inden fazlasını kazandı. İki değerlendirme birbirini tamamlar: değer ağı güçlü ama yavaş bir RL politikasını yaklaşır, rollout zayıf ama hızlı bir politikanın kesin sonucunu verir.

Tek makinelik AlphaGo (48 CPU, 8 GPU), öteki Go programlarına karşı 495 oyunun 494'ünü (%99.8) kazandı. Dağıtık sürüm 1202 CPU ve 176 GPU kullandı.

### AlphaGo Zero: tek ağ, sıfırdan {#zero}

AlphaGo Zero (Silver ve ark. 2017) bu yapıyı radikal biçimde sadeleştirdi:

- **İnsan verisi yoktur.** Rastgele ağırlıklarla, yalnız kurallarla başlar.
- **Elle tasarlanmış öznitelik yoktur.** Girdi $19\times19\times17$'dir: iki oyuncunun son 8 konumdaki taşları ve sıranın kimde olduğu.
- **Tek ağ vardır.** Bir artık ağ (ResNet, 20 ya da 40 blok), politika ve değeri birlikte verir: $(\mathbf p,v)=f_\theta(s)$.
- **Rollout yoktur.** Yaprak yalnız $v$ ile değerlendirilir.

Döngü şöyle işler (Şekil 1, alt sıra):

1. **Kendine karşı oyun.** En iyi ağ $f_{\theta^*}$ ile her hamlede 1600 simülasyonluk MCTS yapılır (hamle başına ≈ 0.4 s). Kökte ziyaret sayılarından bir politika çıkar:
   $$\boldsymbol\pi(a\mid s)=\frac{N(s,a)^{1/\tau}}{\sum_bN(s,b)^{1/\tau}}.$$
   İlk 30 hamlede $\tau=1$ kullanılır ve hamle $\boldsymbol\pi$'den örneklenir; açılışlar çeşitlenir. Sonra $\tau\to0$, yani en çok ziyaret edilen hamle seçilir. Örnek: $N=(10,30,60)$ ve $\tau=0.5$ için $N^2=(100,900,3600)$ olur ve $\pi(a_3)=3600/4600=0.783$'tür.
2. **Eğitim.** Her konum için $(s,\boldsymbol\pi,z)$ kaydedilir; $z$ oyunun sonucudur, o konumda sırası gelen oyuncunun bakışıyla. Ağ şu kaybı en küçükler:
   $$l=(z-v)^2-\boldsymbol\pi^\top\log\mathbf p+c\|\theta\|^2,\qquad c=10^{-4}.$$
   Örnek: $z=1$, $v=0.5$, $\boldsymbol\pi=\mathbf p=(\tfrac12,\tfrac12)$, $c=0$ ise $l=0.25+\ln2=0.943$.
3. **Değerlendirici.** Her 1000 eğitim adımında yeni bir ağ çıkar. Yeni ağ en iyi ağa karşı 400 oyun oynar ($\tau\to0$). Yeni ağ en iyi ağ olur, eğer:
   - oyunların %55'inden fazlasını kazanırsa;
   - eşik 0.5 değildir, çünkü gürültü yüzünden yanlış kabul edilmesin.

:::türetim Arama bir politika iyileştirme operatörüdür
M11.3'teki politika yinelemesinin iki adımı vardır: değerlendirme ve iyileştirme. AlphaGo Zero'nun döngüsü bunun örneklemeli biçimidir:

- **İyileştirme.** Ağın önseli $\mathbf p$ ile başlayan MCTS, değer ağının değerlendirmelerini birleştirerek $\mathbf p$'den **daha güçlü** bir $\boldsymbol\pi$ üretir. Arama bir politika iyileştirme operatörüdür. Kayıptaki $-\boldsymbol\pi^\top\log\mathbf p$ terimi, ağın politikasını bu iyileşmiş politikaya doğru çeker (bir projeksiyon).
- **Değerlendirme.** Oyunun sonucu $z$, iyileşmiş politikanın değerinin bir Monte Carlo örneğidir. $(z-v)^2$ terimi değer başlığını bu politikanın değerine çeker.

İyileşen ağ daha iyi önseller ve değerler verir, onlarla yapılan arama daha da güçlenir. Döngü, aramanın "sabit noktasında" durur: ağın politikası aramanın politikasına eşit olduğunda arama artık iyileştirme sağlamaz. Yaklaşık politika yinelemesi (M11.3) ve Anthony ve ark.'nın (2017) "uzman yinelemesi" bu yorumun kuramsal karşılıklarıdır. Aramasız kendine karşı oyun ise bu operatörü kaybeder. Hedef $\boldsymbol\pi=\mathbf p$ olur ve politika yerinde sayar. Aşağıdaki mini AlphaZero deneyi bunu doğrudan ölçüyor.
:::

**Keşif.** Kendine karşı oyunda iki kaynak vardır. İlki ilk 30 hamledeki $\tau=1$ örneklemesidir. İkincisi kökteki **Dirichlet gürültüsüdür**:
$$P(s_0,a)=(1-\epsilon)p_a+\epsilon\eta_a,\qquad\boldsymbol\eta\sim\text{Dir}(0.03),\ \epsilon=0.25.$$
Örnek: $p=0.4$, $\eta=0.2$ için $P=0.75\cdot0.4+0.25\cdot0.2=0.35$. Gürültü yalnız kökte ve yalnız kendine karşı oyunda eklenir; değerlendirme maçlarında kapatılır. Gürültünün işlevi şudur: ağın neredeyse sıfır olasılık verdiği bir hamle gerçekten iyi olsa bile, PUCT bonusu önselle orantılı olduğu için o hamle nadiren denenir. Ağ onu hiç denemediği için öğrenemez; kendini besleyen bir kör nokta oluşur. Gürültü her oyunda rastgele birkaç hamleye önsel ekler, ama arama kötü hamleleri yine reddedebilir. Dirichlet'in küçük $\alpha$'sı gürültüyü birkaç hamleye yığar ($\alpha\approx10/K$, Go'da $K\approx361$).

::lab l1

**Diğer ayrıntılar:**

- **Simetri.** Go tahtası 8 dihedral simetriye sahiptir (4 döndürme × 2 yansıma). AlphaGo Zero eğitim verisini bu simetrilerle artırdı ve aramada her yaprağı rastgele bir simetriyle değerlendirdi.
- **Teslim olma.** Kök ve en iyi çocuk değerleri bir eşiğin altına inince oyun bırakılır. Eşik, kazanılabilecek oyunların bırakılma oranı (yanlış pozitif) %5'in altında kalacak biçimde otomatik seçilir. Bunu ölçmek için oyunların %10'unda teslim olma kapatılır.
- **Kaynak.** Ölçek, öncüllere göre çok küçüktür. Kendine karşı oyun 4 TPU'lu tek makinede yapılır. Eğitim 64 GPU işçisiyle, 2048'lik yığınlarla yapılır.

**Sonuçlar.**

| sürüm | eğitim | sonuç |
|---|---|---|
| Zero, 20 blok | 3 gün, 4.9 milyon oyun | 36 saatte AlphaGo Lee'yi geçti, 72 saatte onu 100–0 yendi |
| Zero, 40 blok | 40 gün, 29 milyon oyun | Elo 5185; AlphaGo Master'ı 89–11 yendi |

Ham ağ, hiç arama yapmadan Elo 3055'e ulaşır. Bu, AlphaGo Fan'ın (aramayla) 3144'üne yakındır. Arama, ağın üstüne 2000 Elo'dan fazla ekler. Ayrı politika ve değer ağları yerine tek ağ kullanmak ve evrişimli ağ yerine artık ağ kullanmak her biri belirgin Elo kazancı sağladı. Tek ağın iki başlığı aynı temsili paylaşır: değer hedefi, politika için de yararlı öznitelikler öğrenilmesini sağlar.

**AlphaZero** (Silver ve ark. 2018) aynı algoritmayı satranç, shogi ve Go'ya hiç değiştirmeden uyguladı. İki sadeleştirme daha yaptı:

- Değerlendirici kapısı kaldırıldı; ağ sürekli güncellenir ve her zaman en son ağ oynar.
- Simetri artırımı kaldırıldı, çünkü satranç ve shogi simetrik değildir.

MuZero (2020) kuralları da bilmez: oyun dinamiğini gizli bir uzayda öğrenir ve onunla planlar (M11.10).

### Elo ve Bradley–Terry {#elo}

Oyuncuların gücü, maç sonuçlarından **Elo** ölçeğiyle kestirilir. Bradley–Terry modeli, $a$'nın $b$'yi yenme olasılığını puan farkının lojistik fonksiyonu olarak varsayar:
$$P(a\text{ yener }b)=\frac1{1+10^{-(e_a-e_b)/400}}=\sigma\bigl(c_{\text{elo}}\ln10\,(e_a-e_b)\bigr),\qquad c_{\text{elo}}=1/400.$$

- 200 puan fark → 0.76 beklenen skor;
- 230 puan → 0.79 (AlphaGo 2016 makalesinin verdiği örnek);
- 400 puan → 0.91.

Puanlar, bütün maç sonuçlarının olabilirliğini en çoklayarak bulunur. Bu bir lojistik regresyondur (her maç bir örnek, öznitelik iki oyuncunun göstergelerinin farkı). AlphaGo makaleleri BayesElo programını kullandı. Hunter'ın (2004) MM algoritması bunu yinelemeli olarak çözer. Ölçeğin sıfır noktası keyfidir. Makalelerde Fan Hui'nin o tarihteki puanına (2908) bağlandı.

**Kestirimin belirsizliği.** $n$ oyunluk bir maçta skor $s$'nin standart hatası $\sqrt{s(1-s)/n}$'dir; beraberlik varsa daha küçüktür. Elo'ya delta yöntemiyle çevrilir: $\partial e/\partial s=400/(\ln10\cdot s(1-s))$. 400 oyunda $s\approx0.55$ için skorun standart hatası 0.025 ve Elo'nunki ≈ 17 puandır. AlphaGo Zero'nun 400 oyunda %55 kapısı, gerçek farkı ≈ 35 puan olan bir ağı yaklaşık yarı yarıya kabul eder. Farkı 0 olan bir ağı ise ≈ %2 olasılıkla yanlışlıkla kabul eder.

<figure class="fig wide"><svg viewBox="0 0 640 240" role="img" aria-label="AlphaGo sürümlerinin Elo puanları ve Elo farkından hesaplanan beklenen skorlar"><text x="14" y="16" class="sm">Elo (Silver ve ark. 2017, Şekil 6) ve AlphaGo Zero'nun beklenen skoru</text><path d="M170.0,26 L170.0,176" class="mute"/><text x="170.0" y="190" text-anchor="middle" class="sm">0</text><path d="M225.55555555555554,26 L225.55555555555554,176" class="mute"/><text x="225.55555555555554" y="190" text-anchor="middle" class="sm">1000</text><path d="M281.1111111111111,26 L281.1111111111111,176" class="mute"/><text x="281.1111111111111" y="190" text-anchor="middle" class="sm">2000</text><path d="M336.6666666666667,26 L336.6666666666667,176" class="mute"/><text x="336.6666666666667" y="190" text-anchor="middle" class="sm">3000</text><path d="M392.2222222222222,26 L392.2222222222222,176" class="mute"/><text x="392.2222222222222" y="190" text-anchor="middle" class="sm">4000</text><path d="M447.77777777777777,26 L447.77777777777777,176" class="mute"/><text x="447.77777777777777" y="190" text-anchor="middle" class="sm">5000</text><text x="164" y="44" text-anchor="end" class="sm">AlphaGo Zero (40 blok)</text><rect x="170" y="32" width="288.1" height="16" class="ca-f"/><text x="462.1" y="44" class="sm">5185</text><text x="164" y="74" text-anchor="end" class="sm">AlphaGo Master</text><rect x="170" y="62" width="269.9" height="16" class="ca-f"/><text x="443.9" y="74" class="sm">4858</text><text x="164" y="104" text-anchor="end" class="sm">AlphaGo Lee</text><rect x="170" y="92" width="207.7" height="16" class="ca-f"/><text x="381.7" y="104" class="sm">3739</text><text x="164" y="134" text-anchor="end" class="sm">AlphaGo Fan</text><rect x="170" y="122" width="174.7" height="16" class="ca-f"/><text x="348.7" y="134" class="sm">3144</text><text x="164" y="164" text-anchor="end" class="sm">ham ağ (aramasız)</text><rect x="170" y="152" width="169.7" height="16" class="ca-f"/><text x="343.7" y="164" class="sm">3055</text><text x="520" y="40" class="sm">Zero'nun beklenen</text><text x="520" y="54" class="sm">skoru:</text><text x="520" y="74" class="sm">Master: 0.868</text><text x="520" y="92" class="sm">Lee: 1.000</text><text x="520" y="110" class="sm">Fan: 1.000</text><text x="520" y="128" class="sm">ham ağ: 1.000</text><text x="14" y="210" class="sm">E(d) = 1/(1 + 10<tspan baseline-shift="4" font-size="75%">−d/400</tspan>); 200 puan → 0.76, 230 puan → 0.79.</text><text x="14" y="228" class="sm">Ölçülen: Zero (20 blok, 3 gün) Lee'yi 100–0, Zero (40 blok) Master'ı 89–11 yendi; E(327) = 0.87.</text></svg><figcaption>Şekil 2 — Elo puanları makaleden; beklenen skorlar formülle hesaplandı. Master'a karşı ölçülen 89–11 (0.89), 327 puanlık farkın öngördüğü 0.87'ye yakın.</figcaption></figure>

::lab l2

### Mini AlphaZero {#mini}

Bütün döngü küçük bir oyunda tarayıcıda kurulabilir. Aşağıdaki deneyde "ağ" bir tablodur: tic-tac-toe'nun her konumu için bir politika $\mathbf p$ (başta yasal hamlelere düzgün) ve bir değer $v$ (başta 0). Kendine karşı oyunun her hamlesinde şu yapılır:

1. Tabloyla PUCT araması yapılır (yaprakta tablo değeri, oyun sonunda kesin sonuç, kökte Dirichlet gürültüsü).
2. Ziyaret sayılarından $\boldsymbol\pi$ çıkar; ilk 4 hamlede ondan örneklenir.

Oyun bitince ziyaret edilen her konumda tablo güncellenir: $\mathbf p\leftarrow\mathbf p+\eta(\boldsymbol\pi-\mathbf p)$, $v\leftarrow v+\eta(z-v)$. Bu, kaybın gradyan adımının tablo karşılığıdır. Ölçüt, tablonun açgözlü hamlesinin minimax'a göre en iyi olup olmadığıdır.

Sonuç, iyileştirme operatörünün rolünü açıkça gösterir. 1000 oyunda kendine karşı oyunda görülen konumlarda tablonun doğru hamle oranı şöyledir:

- arama yokken ($S=0$) %53'te kalır;
- 25 simülasyonluk aramayla %91'e çıkar;
- 3000 oyun ve 50 simülasyonla %98'e çıkar.

Tablonun sınırı da görülür. Bütün 4520 erişilebilir konum üzerinden oran çok daha düşüktür, çünkü iyi oyunda hiç ortaya çıkmayan konumlar hiç güncellenmez. Sinir ağının genellemesi tam da bu boşluğu doldurur.

::lab l3

### Özet {#ozet}

- Go'nun zorluğu genişlik ($b\approx250$) ve derinliktir ($d\approx150$). Politika genişliği daraltır, değer derinliği keser.
- AlphaGo: SL politika $p_\sigma$ (KGS, %57.0), hızlı rollout $p_\pi$ (%24.2, 2 µs), kendine karşı oyunla REINFORCE'lu $p_\rho$ (oynanan hamle × $z$), her biri ayrı oyundan 30 milyon konumla değer ağı $v_\theta$ (aşırı uyumdan kaçınma), PUCT'lu arama, yaprakta $(1-\lambda)v_\theta+\lambda z$ ($\lambda=0.5$ en iyi). Önselde SL ağı RL ağından iyidir (çeşitlilik).
- Slaytlardaki RL gradyanı uzman hamlesini, değer güncellemesi yanlış işareti kullanıyor; düzeltildi.
- AlphaGo Zero: insan verisi ve rollout yok, tek ResNet $(\mathbf p,v)=f_\theta(s)$, kayıp $(z-v)^2-\boldsymbol\pi^\top\log\mathbf p+c\|\theta\|^2$, $\boldsymbol\pi\propto N^{1/\tau}$, kökte Dir(0.03) gürültüsü, 400 oyunda %55 kapısı. Arama politika iyileştirme, $z$ değerlendirme operatörüdür.
- Sonuçlar: Zero AlphaGo Lee'yi 72 saatte 100–0, 40 günlük sürümü Master'ı 89–11 yendi (Elo 5185). AlphaZero aynı algoritmayı satranç ve shogiye taşıdı.
- Elo: $P=1/(1+10^{-d/400})$; puanlar lojistik olabilirlikle kestirilir. Kestirimin belirsizliği oyun sayısıyla küçülür ve kabul kapılarının yanlış kabul oranını belirler.
- Mini AlphaZero: arama olmadan kendine karşı oyun politikayı iyileştirmez; arama ile tablo hızla en iyi oyuna yaklaşır.
