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

### Etkileşimden öğrenmek: problem, yöntem, alan {#sutton-problem}

Bir bebek kollarını oynatarak, bir sürücü direksiyonu çevirerek, konuşan biri karşısındakinin tepkisini izleyerek neden–sonuç ilişkileri öğrenir. Her hareket için doğru cevabı söyleyen bir öğretmen gerekmez. Sutton ve Barto bu fikri önce ideal bir hesaplama problemi olarak ele alır: bilimsel veya ekonomik bir amacı gerçekleştiren makineler tasarla; önerilen yöntemi matematik ve deneyle değerlendir. Hayvanların nasıl öğrendiğini doğrudan açıklamak ayrı bir araştırma sorusudur; psikoloji ve sinirbilim bağlantıları kitabın Bölüm 14–15 konusudur.

**Pekiştirmeli öğrenme** adı üç anlamda kullanılır: uzun vadeli ödülü artırma **problemi**, bu problemi çözen **yöntemler** ve bunları inceleyen **alan**. Problemle yöntemi ayır: bir MDP tamamen biliniyorsa dinamik programlama ile çözmek de aynı karar problemini çözmektir. Deneme–yanılma ve gecikmiş sonuçlar, deneyimden öğrenen yöntemlerin iki belirleyici özelliğidir. Yazarlar RL alanını, uzun vadeli amaca ulaşmak için etkileşimden öğrenmenin hesap sorunlarını ciddi biçimde ele alan ilk alan olarak değerlendirir; bu, yazarların tarihsel görüşüdür. Kitabın formel çerçevesi, eksik bilinen MDP’lerin optimal kontrolüdür (kaynak Bölüm 3). Duyum, eylem ve amaç MDP modelinde birlikte temsil edilir; ayrıntılı matematiği aşağıda ve M11.2 içinde kuruyoruz.

Denetimli öğrenmede dışarıdan gelen etiket, örnekteki doğru cevabı belirtir ve görülmemiş örneklere genelleme amaçlanır. Etkileşimde doğru ve temsil edici etiketleri her durum için toplamak çoğu kez mümkün değildir. Denetimsiz öğrenme etiketsiz veride yapı arar. RL etiketsiz olduğu için bu ikinci probleme indirgenmez: keşfedilen yapı ancak ödülü artırmaya yardım ederse araçtır. Saf denetimli ve denetimsiz kurgu, ajanın bugün hangi veriyi üreteceğini seçmesinden doğan keşif–yararlanma sorununu içermez.

**Keşif** henüz iyi bilinmeyen eylemleri denemek; **yararlanma** iyi görünenleri seçmektir. Sürekli keşif öğrenileni kullanmaz; sürekli yararlanma daha iyi bir seçeneği hiç görmeyebilir. Rastgele ödüllerde bir eylemin beklenen sonucunu anlamak için birden fazla deneme gerekir. Kaynağın “çözülmemiş ikilem” ifadesi bütün özel problemlerin açık olduğu anlamına gelmez: belli varsayımlar altında kanıtlanmış yöntemler vardır; genel ve her ortama uygun tek bir denge yoktur (M11.7).

RL yaklaşımı, belirsizlik içinde işleyen tam bir amaçlı döngüden başlar. Sadece sınıflandırıcı geliştirmek veya hazır modelle plan yapmak, modelin nereden geldiğini ve sonucun gerçek zamanda nasıl kullanılacağını açıklamaz. Alt problemleri ayrı incelemek yararlıdır; bütün ajandaki görevleri de belirtilmelidir. Ajan mutlaka bütün robot değildir: batarya doluluğunu izleyip robotun kontrol sistemine komut veren bir bileşen de ajandır. Onun ortamı, robotun geri kalanı ile dış dünyadan oluşur.

İstatistik ve optimizasyon RL araçlarını; kontrol ve yöneylem araştırması durum, değer ve boyut sorunlarını; psikoloji ve sinirbilim öğrenme hipotezlerini besler. Parametreli değer yaklaşımı büyük durum uzaylarında hesaplamayı azaltabilir; boyut sorununu koşulsuz çözmez. İlişki iki yönlüdür: biyolojiden esinlenen algoritmalar, hayvan öğrenmesi ve beynin ödül sistemi için sınanabilir modeller de üretir. Kaynak, 1960 sonrasındaki çok sayıda özel bilgi ve sezgisel kural biriktirmeye dayalı “güçlü yöntemler” yaklaşımını, arama ve öğrenme gibi “zayıf yöntemler” diye adlandırılan genel ilkelerle karşılaştırır. Bu adlar başarı garantisi değildir. Kitabın savunduğu araştırma yönü, az sayıda genel öğrenme ve karar ilkesinin ne kadarını açıklayabileceğini araştırmaktır.

Kaynak: Sutton ve Barto, 2. baskı, s. 1–4 (§1.1; s. 1 dipnotu dahil).

### Beş örneğin ortak yapısı {#sutton-ornekler}

Kaynağın örnekleri aynı döngüyü çok farklı ölçeklerde gösterir:

| Örnek | Durum, karar ve amaç | Gecikme ve geri bildirim |
|---|---|---|
| Usta satranççı | Konumdan hamleye; kazanma | Rakibin karşılığını ve kendi sonraki hamlesini planlar; öğrenilmiş konum sezgisi de kullanır. |
| Rafineri kontrolü | İşletme parametrelerini gerçek zamanda ayarlamak | Mühendislerin başlangıç ayarlarına bağlı kalmadan, belirtilen marjinal maliyetlerle verim–maliyet–kalite dengesini arar; bugünkü karar depoların sonraki seviyelerini değiştirir. |
| Yeni doğan ceylan | Ayağa kalkmak ve koşmak | Kaynağın betimlemesinde birkaç dakika içinde ayağa kalkmaya çalışır, yarım saat sonra hız 20 mil/saat olur. Bu yaklaşık 32.19 km/saat değeridir; düşmek doğrudan geri bildirimdir. Bu gözlem bütün türler için deneysel bir yasa olarak sunulmaz. |
| Çöp toplayan robot | Yeni odayı araştırmak veya şarja dönmek | Batarya düzeyini, şarjı geçmişte ne kadar kolay ve hızlı bulduğunu dikkate alır; yeni konum ve kalan enerji gelecekteki seçenekleri değiştirir. |
| Kahvaltı hazırlayan kişi | Beslenme amacını alt amaçlara bölmek | Dolaba gitme, açma, gevrek kutusunu seçme, uzanma, kavrama, çıkarma; kâse, kaşık ve sütü edinme; nesneleri masaya hangi sırayla taşıyacağına karar verme. Göz hareketleri bilgi toplar ve eli/bedeni yönlendirir; açlık, bedensel ihtiyaç ve tercihler de duruma dahildir. Sütü dökerken taşmayı gözleyip hareketi düzeltir. |

Her örnekte etkin bir karar verici vardır; ortam belirsizdir; eylemler hem sonraki durumu hem sonraki karar olanaklarını değiştirir. Dolaylı sonuçlar öngörü gerektirirken eksik öngörü sık gözlem ve tepki gerektirir. Amaç, ajanın algılayabildiği ilerleme sinyalleriyle bağlantılıdır: maçı kazanmak, üretim miktarı, düşmek, bataryanın bitmesi veya kahvaltıdan alınan doyum. Deneyim satranç sezgisini, koşu verimini ve kahvaltı hazırlama sırasını iyileştirebilir. Tasarım, evrim veya önceki görevlerden gelen bilgi neyin kolay öğrenileceğini değiştirir; ortamın özel koşullarına uyum yine etkileşim ister. Kahvaltı örneğinin çıkış noktası Agre (1988) çalışmasıdır.

Kaynak: Sutton ve Barto, s. 4–5; kaynak notu s. 22. Birim dönüşümü hesaplandı: $20\times1.609344=32.18688$.

### Politika, ödül, değer ve model {#sutton-bilesenler}

**Politika** algılanan durumdan eyleme veya eylem olasılıklarına gider. Basit bir tablo, bir fonksiyon ya da pahalı bir arama işlemi olabilir; psikolojideki uyarıcı–tepki bağlantısını andırır. Davranışı belirlemek için politika yeterlidir. Politika kuran her sistemin ayrıca açık bir değer tablosu tutması gerekmez.

**Ödül sinyali** ortamın her adımda verdiği tek sayıdır. Sorunun hangi olayları iyi veya kötü saydığını ve uzun vadede neyin artırılacağını belirler. Haz ve acı benzetmesi sezgiseldir. Sinyal, durum ve eylemin rastgele bir fonksiyonu olabilir. Düşük ödül bir politika değişikliğini tetikleyebilir; tek gözlem tek başına eylemin kötü olduğunu kanıtlamaz.

**Değer**, bir durumdan başlayınca gelecekte beklenen toplam ödülü tahmin eder. Anlık ödülü düşük olan bir durum, bol ödüllü durumlara kapı açtığı için yüksek değerli olabilir; tersi de mümkündür. Ödül amaç açısından birincildir, değer bu amacın öngörüsüdür. Eylem seçerken yalnız anlık ödülü değil uzun vadeli değeri dikkate alırız. Ödül doğrudan gözlenir; değeri ise uzun gözlem dizilerinden tekrar tekrar kestirmek gerekir. Bu güçlük, değer öğrenmesini pek çok RL algoritmasının merkezine koyar.

**Ortam modeli**, durum ve eylemden sonraki duruma ve ödüle ilişkin çıkarım yapmayı sağlar. Gerçek deneyimden önce olası gelecekleri değerlendirmek **planlama**dır. Model tabanlı yöntem model kullanır; modelsiz yöntem örneklerle ilerler. M11.10 içindeki Dyna, etkileşimden hem değer hem model öğrenip modelle planlamayı birleştirir. RL bu nedenle düşük düzeyli deneme–yanılmadan uzun vadeli planlamaya uzanır.

Kaynak: Sutton ve Barto, s. 6–7 (§1.3).

### Durum temsili ve politika aramasının sınırları {#sutton-kapsam}

**Durum sinyali**, ajanın o anda ortam hakkında kullanabildiği bilgidir. Politika ve değere girdi, modele hem girdi hem çıktı olur. Markov anlamındaki tam durum ile eksik bir gözlemi ayırmak gerekir. Kaynak karar verme sorununa odaklanmak için durum sinyalini çoğunlukla hazır alır; ön işleme nominal olarak ortamın parçasıdır. Temsilin öğrenilmesini önemsiz saymaz; kitap §17.3 içinde yeniden ele alır.

Genetik algoritma, genetik programlama ve benzetimli tavlama gibi yöntemler değer öğrenmeden doğrudan politika arayabilir. Bir politikayı bir ortam kopyasında uzun süre sabit tut, aldığı ödülle değerlendir; başarılı politikaları ve rastgele varyasyonlarını sonraki nesle taşı. Küçük veya iyi yapılandırılmış politika uzayı, bol hesap zamanı ve tam gözlenemeyen durumlar bu yaklaşımı yararlı kılabilir. Buna karşılık yalnız politika başına toplam başarıyı kullanmak, hangi durumların ziyaret edildiğini ve hangi eylemin ne işe yaradığını atar. Bu ara bilgi, algı yanıltıcı değilse aramayı çok daha verimli yapabilir. Evrim ile yaşam boyunca öğrenme birlikte kullanılabilir; tek başına evrimsel arama, kitabın çevrimiçi etkileşimden öğrenme odağı değildir.

Kaynak: Sutton ve Barto, s. 7–8 (§1.4).

### XOX: kazanma olasılığını öğrenmek {#sutton-xox}

X ve O sırayla boş bir hücreyi doldurur. Üç aynı işaret yatay, dikey veya çapraz dizilirse oyun biter; tahta dolup bu gerçekleşmezse beraberlik olur. Kusursuz oyuncu yenilmez. Buradaki amaç ise **hata yapan sabit bir rakibe karşı kazanma olasılığını** artırmaktır; beraberlik ve yenilgi aynı sonuç olan 0 ile değerlendirilir. Kaynağın numarasız tahta örneğinde X sol üstten sağ alta çaprazı tamamlamıştır:

<figure class="fig wide"><svg viewBox="0 0 640 242" role="img" aria-label="Üç çarpı ile kazanılan kaynak XOX konumu"><text x="20" y="25" class="sm" style="font-size:18px">X çaprazı tamamladı: sonuç = 1</text><path d="M291,42 V222" class="ink w2" stroke="currentColor"/><path d="M351,42 V222" class="ink w2" stroke="currentColor"/><path d="M231,102 H411" class="ink w2" stroke="currentColor"/><path d="M231,162 H411" class="ink w2" stroke="currentColor"/><text x="261" y="82" text-anchor="middle" fill="currentColor" style="font-size:25px">X</text><text x="321" y="82" text-anchor="middle" fill="currentColor" style="font-size:25px">O</text><text x="381" y="82" text-anchor="middle" fill="currentColor" style="font-size:25px">O</text><text x="261" y="142" text-anchor="middle" fill="currentColor" style="font-size:25px">O</text><text x="321" y="142" text-anchor="middle" fill="currentColor" style="font-size:25px">X</text><text x="381" y="142" text-anchor="middle" fill="currentColor" style="font-size:25px">X</text><text x="381" y="202" text-anchor="middle" fill="currentColor" style="font-size:25px">X</text><path d="M246,57 L396,207" class="ca w3" stroke="currentColor" fill="none"/></svg><figcaption>Şekil 2 — XOX kuralları, kaynak s. 8 içindeki numarasız tahta. Aynı hücre dizilimiyle kazanan çapraz programla doğrulandı (hesaplandı).</figcaption></figure>

Minimax, rakibin en iyi karşılığını varsayar. Yenilgi riski taşısa bile bu rakibin hatalarıyla hep kazanılan bir konuma girmeyi reddedebilir; dolayısıyla gerçek rakibe karşı kazanmayı en çoklamak farklı bir amaçtır. Rakibin her tahtadaki hamle olasılıkları bilinse dinamik programlama kullanılabilir. Bilinmiyorsa deneyimden model kestirip sonra planlamak da bir seçenektir. Doğrudan politika araması ise bir hamle tablosunu pek çok oyunla değerlendirir; tepe tırmanma veya bir politika popülasyonu ile yeni tablolar dener.

Değer yaklaşımında her tahta için bir $V(s)$ sakla. X kazanmışsa 1, O kazanmışsa veya tahta beraberlikle dolmuşsa 0, diğer konumlarda başlangıç 0.5 olsun. Her aday hamlenin oluşturduğu **hamle sonrası konumu** değerlendir; çoğunlukla en yüksek değeri seç, bazen diğer hamlelerden rastgele seç. Sonuncusu, hiç görülmeyecek konumları görmeye yarayan keşiftir. Kaynağın açıklaması sayısal olarak belirlenmiş bir keşif olasılığı vermez; burada sabit bir sayı kaynağa mal edilmez.

Kaynağın güncellemesi, açgözlü hamleden sonraki değeri önceki kendi hamlemizden sonraki konuma taşır. Aradaki rakip hamlesi böylece bir örnek geçişin parçasıdır. Şekilde $a\to b$ rakibin, $b\to c$ bizim hamlemizdir; $V(c)$ ile $V(a)$ güncellenir. İkinci kendi hamlemiz $d\to e$ keşif olduğu için $V(c)$ bu kez güncellenmez; daha yüksek değerlendirilen alternatif $e^*$ seçilmemiştir. Sonraki açgözlü $f\to g$ hamlesi $V(e)$ değerini $V(g)$ yönüne taşır. Siyah düz çizgiler oynanan, kesikli dallar değerlendirilen ama oynanmayan hamlelerdir; kırmızı oklar öğrenmeyi gösterir.

<figure class="fig wide"><svg viewBox="0 0 640 430" role="img" aria-label="a ile g arasındaki XOX hamleleri, keşif ve iki geriye öğrenme oku"><defs><marker id="sutton-ch1-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#d34c4c"/></marker></defs><text x="300" y="17" text-anchor="middle" class="sm" style="font-size:18px">başlangıç konumu</text><path d="M300,35 L300,87" stroke="currentColor" fill="none"/><text x="19" y="65.0" class="sm" style="font-size:18px">rakibin hamlesi</text><path d="M170,43 Q160,43 165,61.0 Q160,79 170,79" class="ink w2" stroke="currentColor" fill="none"/><path d="M300,87 L258,139" stroke="currentColor" fill="none"/><text x="19" y="117.0" class="sm" style="font-size:18px">bizim hamlemiz</text><path d="M170,95 Q160,95 165,113.0 Q160,131 170,131" class="ink w2" stroke="currentColor" fill="none"/><path d="M258,139 L258,191" stroke="currentColor" fill="none"/><text x="19" y="169.0" class="sm" style="font-size:18px">rakibin hamlesi</text><path d="M170,147 Q160,147 165,165.0 Q160,183 170,183" class="ink w2" stroke="currentColor" fill="none"/><path d="M258,191 L387,243" stroke="currentColor" fill="none"/><text x="19" y="221.0" class="sm" style="font-size:18px">bizim hamlemiz</text><path d="M170,199 Q160,199 165,217.0 Q160,235 170,235" class="ink w2" stroke="currentColor" fill="none"/><path d="M387,243 L387,295" stroke="currentColor" fill="none"/><text x="19" y="273.0" class="sm" style="font-size:18px">rakibin hamlesi</text><path d="M170,251 Q160,251 165,269.0 Q160,287 170,287" class="ink w2" stroke="currentColor" fill="none"/><path d="M387,295 L451,347" stroke="currentColor" fill="none"/><text x="19" y="325.0" class="sm" style="font-size:18px">bizim hamlemiz</text><path d="M170,303 Q160,303 165,321.0 Q160,339 170,339" class="ink w2" stroke="currentColor" fill="none"/><path d="M300,87 L200,133" class="dash" stroke="currentColor" fill="none"/><path d="M300,87 L354,133" class="dash" stroke="currentColor" fill="none"/><path d="M300,87 L405,133" class="dash" stroke="currentColor" fill="none"/><path d="M258,191 L158,237" class="dash" stroke="currentColor" fill="none"/><path d="M258,191 L206,237" class="dash" stroke="currentColor" fill="none"/><path d="M258,191 L312,237" class="dash" stroke="currentColor" fill="none"/><path d="M258,191 L363,237" class="dash" stroke="currentColor" fill="none"/><path d="M387,295 L287,341" class="dash" stroke="currentColor" fill="none"/><path d="M387,295 L335,341" class="dash" stroke="currentColor" fill="none"/><path d="M387,295 L492,341" class="dash" stroke="currentColor" fill="none"/><circle cx="300" cy="35" r="3.8" fill="currentColor"/><text x="310" y="40" class="sm" style="font-size:18px">a</text><circle cx="300" cy="87" r="3.8" fill="currentColor"/><text x="310" y="92" class="sm" style="font-size:18px">b</text><circle cx="258" cy="139" r="3.8" fill="currentColor"/><text x="268" y="144" class="sm" style="font-size:18px">c = c*</text><circle cx="258" cy="191" r="3.8" fill="currentColor"/><text x="268" y="196" class="sm" style="font-size:18px">d</text><circle cx="387" cy="243" r="3.8" fill="currentColor"/><text x="397" y="248" class="sm" style="font-size:18px">e</text><circle cx="387" cy="295" r="3.8" fill="currentColor"/><text x="397" y="300" class="sm" style="font-size:18px">f</text><circle cx="451" cy="347" r="3.8" fill="currentColor"/><text x="461" y="352" class="sm" style="font-size:18px">g = g*</text><circle cx="206" cy="237" r="4" stroke="currentColor" fill="none"/><text x="184" y="257" class="sm" style="font-size:18px">e*</text><path d="M258,135 C210,95 242,53 295,37" fill="none" stroke="#d34c4c" stroke-width="1.6" marker-end="url(#sutton-ch1-arrow)"/><path d="M451,343 C525,286 468,247 393,243" fill="none" stroke="#d34c4c" stroke-width="1.6" marker-end="url(#sutton-ch1-arrow)"/><text x="420" y="379" class="sm" style="font-size:18px">… devam edebilir</text><text x="19" y="412" class="sm" style="font-size:18px">düz: oynanan · kesikli: denenmeyen · kırmızı: değer aktarımı</text></svg><figcaption>Şekil 3 — Sutton ve Barto, Şekil 1.1 ilişkilerinin yeniden kurulması: c → a, g → e; e keşif hamlesi, e* daha yüksek değerli alternatif. Dallar yasal oyun ağacı üzerinde denetlendi; bir örnek yörünge gerçek kazanma olasılıklarıyla seçildi (hesaplandı). Geometri kaynak görselden alınmadı.</figcaption></figure>

:::türetim Zamansal fark güncellemesi
Önceki kendi hamlemizden sonraki konum $S_t$, sonraki açgözlü hamlemizden sonraki konum $S_{t+1}$ olsun. Tahmin farkı $\delta_t=V(S_{t+1})-V(S_t)$ ile
$$V(S_t)\leftarrow V(S_t)+\alpha\,[V(S_{t+1})-V(S_t)]
=(1-\alpha)V(S_t)+\alpha V(S_{t+1}).$$
$0<\alpha\le1$ için eski tahmin ve hedef arasında ağırlıklı ortalamadır. Örneğin $V(S_t)=0.5$, hedef $0.8$, $\alpha=0.1$ ise yeni değer $0.53$ olur (hesaplandı). Zamansal fark adı, farklı zamanlarda aynı başarı olasılığına ilişkin iki kestirimin farkından gelir; ayrıntılı yöntem M11.4 içindedir.
:::

Sabit rakip, yeterli ziyaret/keşif ve uygun azalan adım büyüklükleriyle böyle tablosal yöntemlerin en iyi oyunun kazanma olasılıklarına yakınsaması beklenir. Sadece “adımı küçültmek” tek başına yeterli koşul değildir; örnekleme koşulları da gerekir. Adımı sıfıra götürmemek, yavaş değişen rakibi izlemeyi sağlayabilir; o zaman gürültü ve izleme arasında denge kurulur.

Politika başına kazanma sıklığı, sabit politika ve sabit rakip için yansız bir başarı kestirimidir. Ancak bütün oyunun başarısını tüm politikaya yükler; hiç oynanmayan durumların kuralları bile aynı toplam puanı paylaşır. Değer yöntemi oyun içindeki konumları ayrı değerlendirir. İkisi de politika uzayında arar; ikincisi ara deneyimi kullanır. Böylece rakibi modellemeden ve açık bir gelecek ağacı aramadan, kısa görüşlü rakibe çok hamleli tuzak kurma gibi planlamanın etkilerini öğrenebilir.

Örnek RL kapsamını daraltmasın: dış rakipsiz görevler, bitmeyen davranış, her adımda farklı büyüklükte ödül ve sürekli zaman da mümkündür; sürekli zamanın matematiği kitapta geliştirilmez. Büyük hatta sonsuz durum uzayları genelleme ister. Tesauro (1992, 1995) yaklaşık $10^{20}$ tavla durumu için değer öğrenmesini sinir ağıyla birleştirerek TD-Gammon geliştirdi; program önceki programları ve daha sonra en iyi insan oyununu aşan düzeye ulaştı (kaynak §16.1). Her durumu görmek mümkün olmadığından benzer konumlardan bilgi taşımak belirleyicidir. Denetimli öğrenme araçları burada işe yarar; sinir ağları tek ve her zaman en iyi seçenek değildir (M11.6).

Sıfırdan başlamak zorunlu değildir: öznitelik, politika parametrelemesi ve ödül tasarımı ön bilgi taşıyabilir (kaynak §9.5, §13.1, §17.4). Tam gözlenmeyen veya aynı görünen durumlar da ele alınabilir. Bu oyuncu, kendi tek hamlesinin tahtayı nasıl değiştireceğini bildiği için bu dar anlamda bir model kullanır; rakibin davranışını modellemediği için ona göre modelsizdir. Hiç tek adımlık modeli olmayan yöntemler de vardır. Model yeterince doğru kurulamıyorsa modelsizlik yararlı olabilir; modelsiz öğrenme model tabanlı yöntemlerin içinde bile bir yapı taşıdır. Eylem, basit bir motor komutu veya karmaşık bir alt problem çözücüsü olabilir; hiyerarşik sistemlerde birkaç düzey eşzamanlı öğrenebilir.

Kaynak: Sutton ve Barto, s. 8–12 (§1.5), Şekil 1.1; s. 22 açıklaması.

### Ağ yaklaşımına kaynak bağlantısı {#sutton-ag-atif}

:::uyarı Kaynak düzeltmesi (Sutton ve Barto, s. 11): sinir ağı bölüm numarası
Girişte ağlar ve derin öğrenme için §9.6 yönlendirmesi veriliyor. Aynı baskının §9.6 başlığı adım büyüklüğünü elle seçmek (s. 222), sinir ağı yaklaşımının başlığı ise §9.7 (s. 223). Bu nedenle ağlarla genelleme bağlantısı §9.7 olarak okunmalı. Denetim, yüklenen PDF içindeki bölüm başlıklarıyla yapıldı.
:::

### Keşif hamlelerinden öğrenince hangi değer değişir? {#sutton-kesif-degeri}

Sadece açgözlü hamlelerden öğrenme, devamında açgözlü davranılan hedefin değerini kestirmeye yönelir. Keşif dahil bütün hamleleri örneklemek ise **gerçek keşifli davranış politikasının** başarısını kestirir. Devam eden hataları hesaba kattığında daha güvenli bir yol tercih edilebilir. Bu ayrım, tek başına “biri her zaman daha çok kazanır” sonucu vermez; keşif oranı, rakip ve politikanın nasıl iyileştirildiği önemlidir. Ayrıntılı tartışma, kaynaktaki Alıştırma 1.4 kartındadır.

Bu deney kaynakta olmayan, ayrımı tam hesapla görünür kılan küçük bir karar ağacıdır. Güvenli dal tek adımda 0.8 olasılıkla kazandırır. Riskli dalın sonraki iki hamlesi kesin kazanç veya kayıptır. Keşif oranı $\varepsilon=0.3$ iken riskli dalın gerçek değeri $0.7$ olur. Açgözlü devam değerini kullanıp kökte riskliyi seçen davranışın başarısı $0.7\cdot0.7+0.3\cdot0.8=0.73$; güvenliyi önceleyen keşifli davranışın başarısı $0.7\cdot0.8+0.3\cdot0.7=0.77$ olur (hesaplandı). Deney XOX öğrenme eğrisinin benzetimi değildir; iki değer hedefinin farkını yalıtır.

::lab l3

Kaynak bağlantısı: Sutton ve Barto, s. 12–13, Alıştırma 1.4.


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

<figure class="fig wide"><svg viewBox="0 0 640 180" role="img" aria-label="Eylemleri yer değiştirmiş kısa koridor: üç durum ve hedef, ikinci durumda sağ ve sol ters çalışır"><rect x="40" y="50" width="110" height="44" rx="6" class="soft-a"/><text x="95" y="77" text-anchor="middle" class="sm">1 (başlangıç)</text><rect x="170" y="50" width="110" height="44" rx="6" class="soft-b"/><text x="225" y="77" text-anchor="middle" class="sm">2 (ters!)</text><rect x="300" y="50" width="110" height="44" rx="6" class="soft-a"/><text x="355" y="77" text-anchor="middle" class="sm">3</text><rect x="430" y="50" width="110" height="44" rx="6" class="soft-p"/><text x="485" y="77" text-anchor="middle" class="sm">hedef</text><text x="40" y="128" class="sm">1: sağ → 2, sol → 1 (duvar).  2: sağ → 1, sol → 3 (eylemler ters).  3: sağ → hedef, sol → 2.  Her adım −1.</text><text x="40" y="150" class="sm">Tüm durumlarda özellik aynı: politika tek bir sayıdır, p = P(sağ). Kesin değer: J(0.5) = −12, J(0.59) = −11.66 (en iyi), J(0.3) = −16.2;</text><text x="40" y="168" class="sm">p = 0 ya da 1 olan her deterministik politika hedefe hiç varmaz (J = −∞). En iyi politika stokastiktir.</text></svg><figcaption>Şekil 4 — Sutton ve Barto'nun (2018) Örnek 13.1'i. Ajan durumları ayırt edemez (aynı özellik), bu yüzden değer tabanlı yöntemlerin açgözlü (deterministik) politikaları başarısız olur. Politika gradyanı stokastik politikayı doğrudan öğrenir ve p* ≈ 0.59'a yakınsar (l2).</figcaption></figure>

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

### RL tarihinin üç hattı {#sutton-tarih}

Kaynak §1.7, modern RL tarihini üç hattın birleşmesi olarak okur: **deneme–yanılma**, **optimal kontrol/dinamik programlama** ve **zamansal farkla tahmin**. Deneme–yanılma hattı hayvan öğrenmesi psikolojisinden erken AI çalışmalarına ve 1980 başındaki canlanmaya uzanır. Optimal kontrol hattı çoğunlukla öğrenme içermez. İlk ikisi uzun süre ayrı gelişmiştir; üçüncüsü önce aralarında kısmi bir bağ kurmuş, 1980 sonlarında birleşme güçlenmiştir. Aşağıdaki tarihler kitabın tarihsel değerlendirmesidir; “ilk” gibi öncelik ifadeleri bu değerlendirmeye atfedilir.

**Kontrol hattı (s. 14–15).** Optimal kontrol adı, bir dinamik sistemin zaman boyunca davranış ölçütünü en iyi yapan kontrolü tasarlamak için 1950 sonlarında yaygınlaştı. Bellman 1950 ortalarında Hamilton–Jacobi düşüncesini durum ve optimal getiri fonksiyonuyla geliştirdi: Bellman denklemi, dinamik programlama (1957a), ayrık stokastik karar problemi olan MDP (1957b). Howard (1960) politika yinelemesini geliştirdi. Durum değişkenlerinin sayısıyla hesap gereksiniminin üstel büyümesi **boyut laneti** adını aldı; genel dinamik programlamanın etkililiği bu maliyeti ortadan kaldırmaz.

Gelişmeler kısmi gözlenebilirlik (Lovejoy, 1991), uygulamalar (White, 1985, 1988, 1993), yaklaşım yöntemleri (Rust, 1996) ve asenkron hesaplamayı (Bertsekas, 1982, 1983) içerir. Sistematik kaynaklar Bertsekas (2005, 2012), Puterman (1994), Ross (1983), Whittle (1982, 1983); optimal kontrol tarihi için Bryson (1996) olarak verilir. Kitabın “genel stokastik optimal kontrolde tek uygulanabilir yol” değerlendirmesi bir tarihsel görüş olarak okunmalıdır; belirli problemler için başka yöntemlerin yokluğu anlamına gelmez.

Disiplinlerin ayrı amaçları, dinamik programlamayı yalnız tam modelle çevrimdışı analitik hesap gibi görmek ve hesabın zamanda geriye, öğrenmenin ileri gitmesi iki hattın bağlantısını geciktirdi. Bellman–Dreyfus (1959) erken bir yakınlaşma; Witten (1977) açık bir birleşmedir. Werbos (1987), 1977 sonrası savunduğu yaklaşımıyla DP ile öğrenmenin sinirsel/bilişsel sistemler açısından bağını vurguladı. Watkins (1989) MDP çerçevesini çevrimiçi öğrenmeyle birleştirdi. Bertsekas–Tsitsiklis (1996) ağlarla DP birleşimine **neurodynamic programming** adını verdi; **approximate dynamic programming** da benzer eksiklikleri aşmaya çalışan bir adlandırmadır. Tam modeli kullanan DP alışılmış anlamda “öğrenmiyor” olsa da kaynak, aynı karar problemini ve birbirine bağlı yinelemeli teorileri birlikte ele alır.

**Etki yasasından bilgisayara (s. 15–17).** Woodworth (1938), deneme–yanılma düşüncesini Bain'in 1850 dönemindeki araştırarak öğrenme anlatımına ve Morgan'ın 1894 hayvan gözlemlerine bağlar. Thorndike'ın **etki yasası** (1911, s. 244), bir durumda doyumla izlenen tepkinin bağının güçlenip yeniden seçilmesini, rahatsızlıkla izlenenin zayıflamasını anlatır. Yasanın ilk biçiminde doyum veya rahatsızlığın büyüklüğü arttıkça bağdaki değişim de büyür. Thorndike ödül/ceza asimetrisi gibi bulgularla yasayı sonra değiştirdi. Ayrıntılı psikoloji bağlantısı kaynak §14.3 içindedir. Tartışmalar için Gallistel (2005), Herrnstein (1970), Kimble (1961, 1967), Mazur (1994); geniş davranış ilkesi yorumu için Hilgard–Bower (1975), Dennett (1978), Campbell (1960), Cziko (1995) verilir. Hull'un teorileri (1943, 1952) ve Skinner'ın deneyleri (1938) bu mirastadır.

Kaynak, hayvan öğrenmesinde “reinforcement” teriminin erken kullanımını Pavlov kitabının 1927 İngilizce çevirisine bağlar: uygun zaman ilişkisi içinde bir uyarıcının davranış örüntüsünü güçlendirmesi. Sonraki kullanımlar zayıflamayı ve uyarıcının kaldırılmasını da kapsadı. Salt dikkat çekmek veya geçici hareketlilik kalıcı öğrenme değildir; etki pekiştirici çekildikten sonra sürmelidir.

Turing'in 1948 taslağında bilinmeyen tepki rastgele seçilip geçici kaydedilir; acı geçici kayıtları siler, haz kalıcılaştırır. Elektromekanik örnekler Ross'un anahtarlarla yolu saklayan labirent makinesi (1933), Walter'ın öğrenen mekanik kaplumbağası (1950, 1951), Shannon'ın zemindeki mıknatıs ve rölelerle yolu hatırlayan Theseus faresi (1951, 1952), Deutsch'un davranış teorisinden model kullanan labirent çözücüsü (1953, 1954; model tabanlı RL ile ortak özellikler için kaynak Bölüm 8) ve Minsky'nin tezindeki değişebilir sinapsı andıran SNARC bileşenleridir (1954; Stochastic Neural-Analog Reinforcement Calculator). Kaynak ek makine bilgisi için cyberneticzoo.com adresini verir.

Farley–Clark (1954) sayısal ağ benzetiminde deneme–yanılma kullandı; Clark–Farley (1955) örüntü tanıma ve genellemeye yöneldi. Rosenblatt (1962) ve Widrow–Hoff (1960) ödül/ceza dilinden esinlense de incelenen algoritmalar doğru cevaplı eğitim örnekleri kullanan denetimli öğrenmeydi. Ağırlıkları hatayla düzeltmek tek başına RL değildir: değerlendirici geri bildirim hangi eylemin doğru olduğunu söylemez. Kaynağa göre bu kavram karışıklığı, 1960–1970 döneminde gerçek deneme–yanılma araştırmasının seyrekleşmesine katkı verdi. Mühendislikte RL terimleri Waltz–Fu (1965), Mendel (1966), Fu (1970), Mendel–McClaren (1970) çalışmalarında görülür. Minsky (1961), tahmin ve beklenti yanında **katkı atama** sorununu öne çıkardı: ortak başarıyı hangi kararlara ne kadar bağlamalı?

**Erken istisnalar ve öğrenen sistemler (s. 17–19).** Andreae'nin STeLLA sistemi model ve gizli durum için “iç monolog” kullandı (1963, 1969a,b); sonraki çalışma öğretmen bilgisini ve yeni olay üretme amacını da içerdi (1977). “Leakback” mekanizması değerleri geriye aktarmayı andırır (1998). Kaynak, etkisinin sınırlı kaldığını belirtir ve 2017a,b değerlendirmelerine yönlendirir. Michie'nin MENACE sistemi (1961, 1963; Matchbox Educable Naughts and Crosses Engine), her XOX konumuna bir kibrit kutusu, her hamleye renkli boncuk atar; rastgele boncuk hamleyi seçer, oyun sonunda kullanılan kutulardaki boncuklar artırılır/azaltılır. Michie–Chambers (1968) GLEE (Game Learning Expectimaxing Engine) ve BOXES geliştirdi. BOXES, araba ucuna ulaşınca veya direk düşünce gelen başarısızlık sinyaliyle denge öğrendi; Widrow–Smith (1964) sürümünde ise dengeyi bilen öğretmen vardı. Bu ayrım Barto–Sutton–Anderson (1983), Sutton (1984) deneylerini etkiledi; Michie (1974) öğrenme ve denemenin AI içindeki rolünü savundu.

Widrow–Gupta–Maitra (1973), LMS kuralını (Widrow–Hoff, 1960) doğru cevap yerine başarı/başarısızlıkla öğrenen **selective bootstrap adaptation** biçimine dönüştürdü ve blackjack uyguladı. “Öğretmenle” yerine “eleştirmenle” öğrenme, buradaki critic teriminin kaynağıdır. Buchanan–Mitchell–Smith–Johnson (1978) ve Dietterich–Buchanan (1984) ise critic adını yalnız değerlendirmeden fazla iş yapan uzman sistem için kullandı.

Tsetlin ve çalışma arkadaşlarının 1960 dönemindeki **öğrenen otomatları** (ölümünden sonra yayımlanan derleme, 1973), $k$ kollu bandit gibi durum–eylem ilişkilendirmesi gerektirmeyen saf seçim problemlerinde düşük bellekle ödül olasılığını artırır. Narendra–Thathachar (1974, 1989), ödüle göre eylem olasılıklarını güncelleyen stokastik biçimleri geliştirdi. Harth–Tzanakou (1974) Alopex (Algorithm of Pattern Extraction), eylem–pekiştirme korelasyonlarını kullandı ve Barto–Sutton–Brouwer (1981) çalışmasını etkiledi. Psikolojideki öncüller Estes (1950), Bush–Mosteller (1955), Sternberg (1963) istatistiksel öğrenme teorileridir.

Cross (1973), Bush–Mosteller yaklaşımını ekonomik modellere taşıdı; Arthur (1991), ideal ekonomik ajandan daha insana benzer karar vericiler amaçladı. Ekonomi hattı AI hattından büyük ölçüde ayrı gelişti ve oyun teorisine uzandı. Camerer (2011) bu geleneği, Nowé–Vrancx–De Hauwere (2012) çok ajanlı genişlemeleri inceler. Buradaki stratejik oyun kuramı, XOX veya dama programına RL uygulamakla aynı araştırma sorusu değildir; ikinci konu için Szita (2012) verilir.

Holland’ın erken çalışmaları bandit ve evrimsel aramadaki gibi durumla ilişkilendirme içermeyen seçim öğrenmesine odaklandı. Holland (1975) seçilime dayalı uyarlanmayı, 1976 ve 1986 içinde durum–eylem ilişkileri ve değerler içeren **classifier systems** yaklaşımını geliştirdi. **Bucket brigade**, katkıyı zamansal fark benzeri şekilde geriye taşır; genetik algoritma yararlı temsil üretir. Urbanowicz–Moore (2009) bu dalı değerlendirir. Kaynak, salt genetik algoritmaları RL saymaz; Fogel–Owens–Walsh (1966) ve Koza (1992) diğer evrimsel hesaplama örnekleridir. Klopf (1972, 1975, 1982), yalnız denetimli öğrenmeye odaklanmanın hedefe yönelme ve istenmeyenden kaçınma gibi hazcı yönleri kaçırdığını savundu. Bu hazcı bakış kaynak §15.9, çok katmanlı ağ bağlantısı §15.10 içinde geliştirilir. Barto–Sutton (1981a) değerlendirmesi yazarların yönünü belirledi. RL/denetimli ayrımını Barto–Sutton–Brouwer (1981), Barto–Sutton (1981b), Barto–Anandan (1985); çok katmanlı ağ öğrenmesini Barto–Anderson–Sutton (1982), Barto–Anderson (1985), Barto (1985, 1986), Barto–Jordan (1987) araştırdı.

**Zamansal fark hattı ve birleşme (s. 20–21).** Birincil pekiştiriciyle eşleşerek benzer etki edinen **ikincil pekiştirici**, gelecekteki başarı tahmininin neden öğrenme sinyali olabileceğine bir köprü kurar. Kaynak, Minsky (1954) erken yorumunu ve ilk uygulama olarak Samuel'in dama programını (1959) vurgular. Samuel'in esini, Shannon'ın çevrimiçi değiştirilebilen satranç değerlendirme fonksiyonu önerisidir (1950); Samuel Minsky'ye atıf yapmadı. Shannon'ın Bellman'ı da etkilediği olasılığı için kaynak kanıt bulunmadığını açıkça söyler. Minsky (1961) Samuel ile ikincil pekiştirmeyi ilişkilendirdi. Kaynağa göre Samuel sonrasındaki yaklaşık on yılda hesaplamalı deneme–yanılma az, hesaplamalı TD çalışması ise görünüşe göre yoktu. Samuel uygulaması kaynak §16.2 içindedir.

Klopf (1972), her nöronun uyarıcı girdiyi ödül, baskılayıcı girdiyi ceza saydığı **genelleştirilmiş pekiştirme** ile alt bileşenlerin birbirini yerel olarak eğitmesini önerdi. Bu, modern TD ile aynı kural değildir; Samuel'in yaklaşımı TD'ye daha yakındır. Sutton (1978a,b,c) ardışık tahmin farklarını geliştirdi; Sutton–Barto (1981a), Barto–Sutton (1982) klasik koşullanma modeli kurdu. Diğer modeller Klopf (1988), Moore ve ark. (1986), Sutton–Barto (1987, 1990); TD olarak yorumlanabilen sinirbilim modelleri Hawkins–Kandel (1984), Byrne–Gingrich–Baxter (1990), Gelperin–Hopfield–Tank (1985), Tesauro (1986), Friston ve ark. (1994) olarak verilir. Matematiksel benzerlik, her örnekte tarihsel etkileşim bulunduğunu göstermez.

Yazarlar Minsky/Samuel bağlantılarını sonradan fark etti; 1981 sürecinde önceki çalışmaların bilincindeyken aktör–eleştirmeni geliştirdi, araba–direğe uyguladı (Barto–Sutton–Anderson, 1983). Sutton (1984) tezinde inceledi; Anderson (1986) geri yayılımlı ağlarla genişletti; Holland (1986) bucket brigade içinde TD düşüncesini kullandı. Sutton (1988), TD'yi kontrolden ayırıp genel tahmin problemi olarak inceledi, TD($\lambda$) ve bazı yakınsama sonuçlarını sundu.

Kaynağın değerlendirmesine göre Witten çalışması, görünüşe göre yayımlanmış en erken TD kuralıdır. Witten (1976a, 1977), ilk kez 1974 içinde dergiye gönderilen ve 1976 tezinde de yer alan uyarlamalı MDP kontrolünde tablosal TD(0) önerdi; STeLLA geleneğinden gelerek üç hattı erken birleştirdi. Yazarlar bu yayını kendi mimarilerini tamamlarken 1981 içinde keşfetti. Watkins'in Q-öğrenmesi (1989) TD ile optimal kontrolü güçlü biçimde birleştirdi; Werbos (1987) bu birlikteliği savunanlar arasındaydı. Tesauro'nun TD-Gammon başarısı (1992) ilgiyi büyüttü. TD sinyali ile dopamin nöronlarının etkinliği arasındaki benzerlik yeni bir sinirbilim hattına dönüştü: Friston ve ark. (1994), Barto (1995a), Houk–Adams–Barto (1995), Montague–Dayan–Sejnowski (1996), Schultz–Dayan–Montague (1997). Ayrıntılar kitabın Bölüm 15 konusudur.

Kaynak: Sutton ve Barto, s. 13–21 (§1.6–1.7). Başka tarihsel değerlendirmeler farklı önceliklendirmeler yapabilir.

### Bölüm 1 için okuma yolları {#sutton-okumalar}

Kaynağın s. 22 kaynak notları, genel RL için Szepesvári (2010), Bertsekas–Tsitsiklis (1996), Kaelbling (1993a), Sugiyama–Hachiya–Morimura (2013) kitaplarını; kontrol/yöneylem bakışı için Si–Barto–Powell–Wunsch (2004), Powell (2011), Lewis–Liu (2012), Bertsekas (2012) çalışmalarını önerir. Cao (2009), stokastik dinamik sistemlerin öğrenme ve optimizasyon yaklaşımlarını karşılaştırır. Machine Learning dergisinin Sutton (1992a), Kaelbling (1996), Singh (2002) özel sayıları; Barto (1995b), Kaelbling–Littman–Moore (1996), Keerthi–Ravindran (1997) taramaları ve Wiering–van Otterlo (2012) derlemesi başka girişlerdir. Kaynak notunda bu son soyadı “Weiring” yazımıyla geçer. Ayrıntılı künyeler kitabın kaynakçasındadır; bu bağlantılar henüz kitabın bütün kaynakçasının aktarımı anlamına gelmez.


### Özet {#ozet}

- Denetimli, yarı denetimli, denetimsiz ve pekiştirmeli öğrenme. RL'de doğru cevap verilmez; ajan ödülü en büyükler. Geri bildirim gecikmeli, ortam stokastik, kararlar birbirine bağlıdır.
- Slayt notları: "Reinforced learning" → reinforcement; Atari ödülü skor değişimidir; $3^{361}\approx1.74\times10^{172}$; "negatif pekiştirme" psikolojide uyaranın kaldırılmasıdır, acı vermek cezadır.
- MDP $(\mathcal S,\mathcal A,P,R,\gamma)$; getiri $G_t=\sum\gamma^kr_{t+k+1}$; $\mathbf V^\pi=(\mathbf I-\gamma\mathbf P_\pi)^{-1}\mathbf r_\pi$.
- Bellman optimallik operatörü $\gamma$-büzülmedir; değer iterasyonu $\|V_k-V^*\|\le\gamma^k\|V_0-V^*\|$ ile yakınsar. El örneği: $V^*=(3,4)$, $Q^*(s_0,\text{kal})=1.5$.
- Politika gradyanı: $\nabla J=\mathbb E[\sum_t\nabla\log\pi(a_t\mid s_t)(G_t-b(s_t))]$; geçiş olasılıkları türevde kaybolur (modelsiz); taban çizgisi yansızdır ve varyansı düşürür. El örneği: 0.5 ve 0.2.
- Durumları ayıramayan ajanda en iyi politika stokastik olabilir (kısa koridor, p* ≈ 0.59, J* ≈ −11.66). Ayrıntılı RL Faz 11'de.

- Sutton ve Barto Bölüm 1: politika, ödül, değer ve model ayrımı; XOX içinde hamle sonrası TD; simetri ve rakip varsayımları; RL tarihinin üç hattı.
