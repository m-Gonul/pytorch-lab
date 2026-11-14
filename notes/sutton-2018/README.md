# Sutton ve Barto (2018): üretim kaydı

Bu kaynak tamamlanmadı. Birim defteri `content/coverage/sutton.yaml` içindedir. `progress.complete: false` kitabın tamamına ilişkindir. İlk işlenen aralık basılı s. 1–22, PDF s. 23–44.

Bölüm 1 şekillerini yeniden hesaplamak için (yalnız Python standart kitaplığı):

```bash
python3 -I notes/sutton-2018/generate_figures.py /tmp/sutton-ch1
```

Çıktılar `board.html`, `tree.html`, `ch1-trace.json`. Tahta kazancı ve yörüngedeki c/g açgözlü seçimleri, e için daha iyi seçilmeyen alternatif bulunması, tam oyun ağacı üzerinde denetlenir. X optimal, O düzgün rastgele varsayılır. Bu, kaynak görselin yasal hamle ve öğrenme ilişkilerini taşıyan özgün bir örnek yörüngedir; kaynakta verilmeyen sayılar ona atfedilmez.

Ek `explorevalue` deneyi farklı bir iki aşamalı ağaçtır; kaynak Alıştırma 1.4 ayrımını açıklamak içindir. XOX eğitim grafiği değildir.

İlk envanterin etiketli sayıları 140 alıştırma, 104 şekil, 38 gri kutulu sözde-kod ve 1 numaralı tablodur. Numarasız görseller ve metin içi algoritmaların bütün kaynak sayımı henüz bitmedi. İşlenmeyen hazır modüller, izin verilen anlamı dışında `bekliyor` olarak işaretlenmedi.

## Bölüm 2

İşlenen ikinci aralık basılı s.25–45 (PDF 47–67). Kaynak eğrileri `suttonbandits` deneyinde yeniden hesaplanır; kaynak parametre çarpanı 1 ve şekil ön ayarları 2000 görev/1000 adımdır. `presets_ch2.json` bu koşulların ölçüm özetidir.

Yoğunluk görselini yeniden üret:

```bash
python3 -I notes/sutton-2018/generate_bandit_density.py /tmp/sutton-density.html
```

Ödev deneylerini, soru aritmetiğini ve sonlu-fark taban denetimini yeniden çalıştır (NumPy/SciPy gerekir; 200000 adımlık gerçek koşu içerir):

```bash
python3 -I notes/sutton-2018/measure_ch2.py /tmp/sutton-measurements.json
```

`measurements_ch2.json`, çalıştırılan 2000×10000 ve 8×200000 koşularının sonuçlarını saklar. İkinci deneyin tekrar sayısı 8, kaynakta verilmiş bir sayı değildir. Her iki kayıt da kullanılan kod ve tohumla değerlendirilmelidir; Monte Carlo sonucu evrensel bir yöntem üstünlüğü kanıtı değildir.

NumPy ile tarayıcının rastgele üreteçleri farklıdır. İkisinde tohum 1 seçmek
aynı ortam ve ödül yörüngeleri üretmez; eşleşen dağılım, parametre ve ölçüm
aralığına rağmen sayısal ortalamaların farklı çıkması beklenir.

## Bölüm 3

Basılı s.47–71 / PDF69–93, sonraki s.72 boş. 179 bilgi birimi, 29 alıştırma, 13 kaynak görseli (5 numaralı/8 numarasız), 10 satırlı numarasız tablo. Bağımsız ikinci okuma yalnız kaynak ve defter üzerinden yapıldı; alt sorular, sayfa aralıkları, tarih ayrıntıları ve şemaların dalları tamamlandı.

Özgün SVG gruplarını ve bağımsız NumPy kontrollerini yeniden üret:

```bash
python3 -I notes/sutton-2018/generate_ch3.py /tmp/sutton-ch3
```

Çıktı f3.html…f8.html ve measurements_ch3.json. Kayıtlı JSON; 25 rastgele V, 25 optimal V, tüm optimal eylem kümeleri, altı robot politikasının değerleri, 3.8/3.9/3.14/3.24 hesapları ve özgün golf modelini içerir. Tarayıcıdaki suttongrid sayıları ayrı JS çözücüsüyle bunlara karşı doğrulandı.

Golfün tam geometrisi ve driver iniş olasılıkları kitapta verilmediğinden 15 durumlu özgün model kullanıldı: 13 mesafe düzeyi ve iki kum durumu. Putter iki düzey ilerler; driver iki olası yere iner. Tee için 6 putt veya 2 drive+putt sayıları hesaplandı; driver için ara kesirli değer de gösterildi. Bu model kaynak şekle ölçülmemiş koordinatlar veya biricik olasılıklar atfetmez. Robotun başlangıç sayısal parametreleri de kaynak sabitleri değil deney seçimidir.

Alıştırma 3.4 tablosu yalnız koşullu beklenen ödüllerle tekil belirlenemez; genel ρ çözümü ve açık belirlenimci-ödül ek varsayımı ayrı verilir. Bellman tekliği sınırlı sonlu model/γ<1 koşuluyla yazılır; γ=1 için karşı örnek ve sonlanma gereksinimi belirtilir. Bölüm 4–17 ve ön/arka materyal henüz tamamlanmadı.

## Bölüm 4

Basılı s.73–90 / PDF95–112. 102 kaynak birimi,10 alıştırma,1 numarasız soru,7 görsel,3 sözde-kod. Ayrı teaching_extensions alanındaki Poisson türetimi kaynak birimi sayılarına katılmaz.

Sayısal bütün referansları yeniden üret (NumPy):

```bash
OPENBLAS_NUM_THREADS=1 python3 -I notes/sutton-2018/measure_ch4.py /tmp/measurements_ch4.json
python3 -I notes/sutton-2018/generate_ch4.py /tmp/sutton-ch4-figures
```

measurements_ch4.json;özgün/değişmiş441 durumlu Jack politikaları ve değerleri,ızgara durumları,kumarbaz0.4/0.25/0.55 tarama izleri ve azalan eşik koşularını saklar. JS çözücüleri bütün değer/politika/izlerle karşılaştırıldı. 0.55 bir-birim bahis için kapalı form(1−r^s)/(1−r^100),r=0.45/0.55 ek bağımsız denetimdir.

Şekil 4.1 kaynak yarımda sıfıra doğru iki anlamlı yuvarlama ile çizilir;hesaplar yuvarlanmaz. Jack sayımları bağımsız Poisson varsayımıyla ve tam doyan kuyruk kütleleriyle kurulur;kaynak tüm ortak bağımlılıkları ayrı belirtmez. Kumarbaz labı yerinde,eski x3 aşaması eşzamanlı günceller. Pozitif bahisler sonlanan politikalar içindir;0 bahis veγ1 nüansı anlatımda açıklanır. GPI izdüşüm şekli bir geometri benzetmesidir,yakınsama kanıtı değildir.

İlk dört bölüm işlendi;Bölüm 5–17 ve ön/arka materyal henüz tamamlanmadı.

Mevcut x3 eşzamanlı kodu da doğrudan 0.25/0.55 ve üç eşikte çalıştırıldı. 10⁻⁶→10⁻¹² en büyük V farkları 1.74812×10⁻⁷ / 0.000189500726; politika değişimi 0. Bunlar yerinde tarama ölçümleriyle karıştırılmaz; x3_synchronous alanında bütün diziler saklıdır.

## Bölüm 5

Basılı s.91–117 / PDF113–139; PDF140 boş. 161 kaynak birimi,14 numaralı alıştırma ve2 numarasız soru,9 görsel(5 numaralı/4 numarasız),5 sözde-kod. M11.4/M11.5 içine eklenir; eski materyal korunur. Bağımsız ikinci okuma27 metinli sayfa ve12 kaynak görüntüsünü yalnız PDF/defter üzerinden denetledi; nihai takipte açık bulgu kalmadı.

Özgün şemalar ve bağımsız NumPy referansları:

```bash
python3 -I notes/sutton-2018/generate_ch5.py /tmp/sutton-ch5
OPENBLAS_NUM_THREADS=1 python3 -I notes/sutton-2018/measure_ch5.py /tmp/ch5-python.json
node notes/sutton-2018/measure_ch5.mjs /tmp/ch5-python.json /tmp/ch5-js.json
```

Kayıtlı `measurements_ch5_python.json`:200 Blackjack durumunun doğrusal değerlendirmesi,bağımsız optimal Q/π,V,krupiye bitiş dağılımları;gerçek10000/500000 Python tahmin bölümü ve1 milyon MC ES bölümü. 100 milyon hedef getirisinin ek denetimi,hesaplanmış terminal sonuç olasılıklarından eşdeğer çokterimli sayım çekilişidir;100 milyon yolu tek tek yürütme iddiası değildir. Kaynak yaklaşık−0.27726,bağımsız model−0.277204191825. İki kaynak yarış geometrisinin bütün yasal eylemli NumPy Bellman çözümleri de aynı dosyadadır.

`measurements_ch5_js.json`:gerçek500000 Blackjack tahmini/1 milyon ES,100×10000 IS Blackjack bölümü,iki pistte50000 MC bölümü,bütün başlangıçlardan gürültüsüz yörüngeler,MC/model politikaları ve değerler. JS/NumPy bütün200 Blackjack değeri/Q,krupiye dağılımları veiki pistin19181 durum değeriyle eşleşir(izler/sıralar farklı olabilir). Sol model/öğrenilmiş başlangıçV−12.112482/−14.458549,sağ−8.951270/−11.447987. Sonlu MC tam optimum diye sunulmaz. Referans tüm yasal eylemleri kullanır;öğrenilmiş hedefte başlangıç çizgisinde iki son hızı0 yapan eylemlerin elenmesi bu iki haritanın bütün optimumV değerlerini değiştirmedi. Bu kendi sürücü tercihimizdir,kaynakta yasaklanan ek bir çevre eylemi diye anlatılmaz. Çizimde hücre köşesi duvar teması çarpışma;bitiş yeşil hücre merkezlerinin düşey doğrusudur. Kaynak segment kesişimi ister,bu geometri ayrıntılarını tekilleştirmez.

Şekil5.4 deneyinde10×100 milyon bölümün kümülatif noktaları,aralık başına pozitif yol-uzunluğu kategorilerinin çokterimli sayımlarıyla **dağılım korunarak** üretilir. Beta/binom özyinelemesi normal yaklaşım kullanmaz. 80 adımdan uzun kuyruk da ayrıca örneklenir ve düşürülmez. Bunun bir milyar yolun açık yürütülmesi olmadığı hem labda hem derste yazılıdır. Öğretim modeli per-decision/discount-aware karşılaştırması20 tekrar×10000 gerçek100 adımlı bölüm kullanır. Sonsuz varyans sonucu grafikten değil geometrik seriyle,ordinary tutarlılığı sonlu birinci moment ve Durrett Teorem2.4.1 ile doğrulanır.

Kaynak s.107 “yakınsamaz” iddiası düzeltilir;MC ES açık-problem ve bilinen weighted-PDIS önerileri hakkındaki ifadeler2018 bağlamında kalır. Fiziksel sabun minimal yüzeyi genel olarak harmonik değildir;tel labı küçük-eğim/ayrık Dirichlet modelidir. Sonlu sabit ufukta ayrı normalize PDIS tutarlılığı kendi öğretim uzantısı olarak kaynak sayımından ayrıdır. Kaynak fotoğrafı ve PDF çizimleri yayımlanmaz;özgün SVG ve hesaplanan yüzeyler kullanılır.

Python doğrulamasında küçük Torch çekirdeklerinin çok fazla iş parçacığı açmasını önlemek için `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` ortamı kullanıldı. GPU/Triton gerektirmeyen iki değişen modül ve notebook gerçek CPU Torch ile denetlendi. Sayıdan bağımsız basit etiketler için üretim tekrarında tüm Monte Carlo koşularını yeniden çalıştırmak gerekmez.

İlk beş bölüm işlendi;Bölüm6–17,ön/arka materyal,kısım ayırıcıları ve kitabın tüm kaynakça/dizin aktarımı tamamlanmadı.

## Bölüm 6

Basılı s.119–140 / PDF141–162. 134 kaynak birimi:14 alıştırma,2 numarasız soru,14 görsel(5 numaralı/9 numarasız),4 kutulu kod,1 tablo ve1 gerçek dipnot. Bağımsız ikinci okuma22 metinli sayfa ve18 görüntüyü yalnız PDF/defter üzerinden denetledi;son takipte açık bulgu kalmadı. M11.4/M11.5 genişletildi;eski aşamalar ve deneyler korunur. İlk6 bölüm841 birimdir;Bölüm7–17 ve ön/arka materyal tamamlanmadı.

Yedi özgün SVG grubunu,aritmetik ve bağımsız modelleri yeniden üret:

```bash
python3 -I notes/sutton-2018/generate_ch6.py /tmp/sutton-ch6
OPENBLAS_NUM_THREADS=1 python3 -I notes/sutton-2018/measure_ch6.py /tmp/ch6-python.json
node notes/sutton-2018/measure_ch6.mjs /tmp/ch6-python.json /tmp/ch6-js.json
```

`measurements_ch6_python.json`,eve dönüşün bütün zaman/ödül/hata dizilerini,6.1/6.3/6.12 hesaplarını,Predictor örneğini,afterstate maliyetini,altı4/8/9 eylemli belirlenimci/stokastik rüzgâr modelinin420 değerini ve bağımsız uçurum MDP çözümlerini içerir. Uçurumda optimal açgözlü V(S)=−13;bu politikanın ε0.1 davranışı−50.8000476668;ε-soft optimum değer−20.7076773987. Bunlar tam model referanslarıdır,sonlu öğrenilmiş tabloların otomatik optimum olduğunun iddiası değildir.

Sabitα ortalama yakınsaması karşı örneği aynı kayıttadır. P=[[.8,.2],[.3,.7]],r=(0,1),γ.9 için gerçekV=(3.27272727,5.09090909);α.1 çevrimiçi Markov güncellemesinin dört birinci-moment denkleminden limiti(3.30319311,5.10246153). 4096 zincir×12000 adım(ilk6000 atılır) gerçek örneklemesi(3.30318634,5.10223722),zincirler arasıSE(.00134294,.00141588). Kaynak2018 s.124 ifadesi,atıf yapılan Sutton1988 Teorem2 içindeki dizi boyunca dondurulmuş/bölüm sonunda güncellenmiş ağırlık koşuluyla ayrılır. Modelin P spektral yarıçapı değil4×4 moment-yineleme matrisininki1 altındadır.

`measurements_ch6_js.json`,eski/yeni bütün mctd,batchmctd,cliffwalk,maxbias ön ayarlarını gerçek JS koşularıyla saklar;her rüzgâr modelinin bütün V değerlerini NumPy ile karşılaştırır(en büyük fark0). Rüzgâr tohum6,8000 adım,α.5/ε.1 koşusu167 bölüm/15 adımlık açgözlü yol buldu. Stokastik8 eylem model optimumu10.9646090535;100000 adımlık öğrenme ardından1000 gerçek değerlendirme bölümü ortalaması13.982(SE.249648),0 bütçe aşımı. Bu iki sayı farklı politikaların değeridir. Tohum1,4 eylem,8000 adım karşı örneğinde son tablonun açgözlü yolu500 adımda bitmez;bu sonuç saklanır. JavaScript ve NumPy eşit sayısal tohumla eşit yollar üretmez.

Kaynak Şekil6.3 tam bütçelerini yeniden çalıştırmak için:

```bash
g++ -O3 -std=c++17 notes/sutton-2018/measure_ch6_cliff.cpp -o /tmp/measure-ch6-cliff
/tmp/measure-ch6-cliff /tmp/ch6-cliff.json 50000 10
```

Bu uzun koşudur:19α×3 yöntem×2 ufuk=114 koşul;ilk100 bölüm her ayarda50000 tekrar,ilk100000 bölüm her ayarda10 tekrar. Toplam342000000 gerçek bölüm ve13385049857 gerçek çevre geçişi yürütüldü;son-pencere ortalamasıyla değiştirilmedi ve bölüm sınırında yapay terminal kullanılmadı. Q0,γ1,ε.1,düzgün rastgele açgözlü bağ çözümü. Kaynak α ekseni açıkça19 nokta0.1…1/adım.05 ile örneklendi. Kayıt `measurements_ch6_cliff.json`;tarayıcı verisi `site/js/data/sutton-ch6-cliff.js` aynı satırları taşır. Satır sırası ilk çalıştırılan onda-birler ve ardından eklenen aradaki noktalar nedeniyle yeniden çalıştırmadan farklı olabilir;her koşul kendi sabit alt-akışını kullandığından değerler değişmez. Grafiğin−160 altı değerleri tabloda ve kayıtta korunur.

En iyi ilk100 ortalamaları SARSAα.75:−68.807829,Qα.9:−73.253825,Expectedα1:−44.1558156. Uzunα1 Q−50.8246/Expected−20.7493 bağımsız model sınırlarıyla uyumludur;sonlu10 tekrarınSE bilgisi kayıttadır. C++ kaydı α/evre kontrolüyle görüntülenir;canlı seed/ε değişimi eski cliffwalk deneyinde yapılır. İki yolun anlamı arayüzde açıklanır.

`cliffwalk` kaynağın gerçek terminal çeşidini varsayılan yapar;eski1000 adım kesmeli denemeler ayrı seçenek olarak kalır ve kesme sayısı bildirilir. SARSA A′ eylemini güncellemeden önce devreder;Q/Expected davranış eylemini güncellemeden sonra seçer. Geometrik düşüş noktası çizimi ve eski mutlak üstünlük/bağımsız çift-tablo iddiaları düzeltildi. Kaynaktaki Weiring adı,birincil2009 makalesine göre Wiering olarak düzeltildi. Kaynak2018 araştırma değerlendirmeleri tarihleriyle korunur.

## Bölüm 7

Basılı s.141–158 / PDF163–180. 83 kaynak birimi:11 alıştırma,7 görsel(5 numaralı/2 numarasız),5 kutulu sözde-kod;tablo/dipnot yok. Bağımsız ikinci okuma18 metin sayfası,17 kaynak görüntüsü ve1 büyütmeyi yalnız PDF/defter üzerinden denetledi. Bütün924 kümülatif hedef doğrulandı;ilk7 bölüm işlendi,8–17 ve ön/arka materyal tamamlanmadı.

Beş özgün SVG grubunu ve bağımsız sayısal kayıtları yeniden üret:

```bash
python3 -I notes/sutton-2018/generate_ch7.py /tmp/sutton-ch7
OPENBLAS_NUM_THREADS=1 python3 -I notes/sutton-2018/measure_ch7.py /tmp/ch7-python.json
node notes/sutton-2018/measure_ch7.mjs /tmp/ch7-python.json /tmp/ch7-js.json
```

Python ve JavaScript ayrı uygulamalardır;bu bölümde **aynı örnek yolları tam karşılaştırmak için** Python tarafında JS Mulberry32 üretecinin bit işlemleri ayrıca yazıldı. Bu özel eşleştirme önceki bölümlerdeki NumPy/random üreteçleri için eşit sayısal tohumun eşit yollar ürettiği anlamına gelmez. Öğrenci x5/x6/x7 ekleri Python `random.Random` kullanır;onların örnekleri bu lab kayıtlarıyla birebir yol eşleştirmesi iddiası taşımaz.

`measurements_ch7_python.json`:kaynak Şekil7.2 koşullarında19durum/sol−1/sağ+1/V0,100 tekrar×ilk10 bölüm;10n×21α=210 koşul. Ortak1000 yürüyüşte96588 gerçek geçiş üretilir ve bütün ayarlarda tekrar kullanılır. αızgarası0…1/adım.05 kendi açık örneklememizdir. Hata önce her bölümün sonundaki19 değerde RMS,sonra bölüm/tekrar ortalamasıdır;son bölüm hatasıyla değiştirilmez. En iyi n4/α.4:0.2683959398,SE.0047680346. Sol0/başlangıç.5 affineşlemesinde bütün210 hata yarıya iner,en büyük dönüşüm farkı1.11×10⁻¹⁶. Başlangıç yine0 kalınca en iyi n16/α.3:0.2701943266. N5,1000 yürüyüş/8944 geçiş koşulunda en iyi n1/α.35:0.2212028739.

7.2 karşılaştırması,her gerçek geçişte güncelleme öncesi saklanan δ hatalarıyla,τ güncellemesinde güncel tablodan hesaplanan n-adım hedefi ayırır. n1 eşleşir;n4 için en iyi saklıδ α.35:0.2733071829,standart hedef α.4:0.2683959398. n4/α.1 saklıδ biraz daha düşük hata verir;bu her ayarda üstünlük değildir. n8/α.8 saklıδ RMS5.9733653×10¹⁹;bu noktalar hesaplardan çıkarılmadı. Birinci karşılaştırma grafiği0…1.2 aralığına odaklanır,ikincisi pozitif hataların tamamını log eksende gösterir. Terminal sonrası flush yeni δ veya yeni çevre adımı üretmez.

7.10 için kendi4durumlu deterministik zincirimiz,b.5/π(.9,.1),γ1,V0,n4,α.1;1000 tekrar×100 bölüm=400000 ortak çevre geçişi. İki eylem aynı sonraki duruma gider,yalnız son ödül1;gerçekV1. Son MSE tam oran6.00438149×10⁻⁷/CV7.18059924×10⁻⁹;bağımsız tekrarlar arasıSE2.6×10⁻⁷/7.6×10⁻¹⁰. İlk MSE≤.001 bölümleri45/39. n1 karşılaştırmasında en büyük fark1.11×10⁻¹⁶;π=b için iki yöntem aynı veMSE=(1−α)^(2E). Durum tekrarı olmadığı için gelecekteki değer tabanları bu bölümün sonraki eylemlerinden bağımsızdır;beklenti ispatı bu açık koşulla yapılır. Kötü veya eylem sonrası seçilen tabanın her zaman varyans azalttığı iddia edilmez.

Kayıt ayrıca3 kaynak düzeltmesinin tam sonuç sayımını içerir:7.11 sonρ faktörüyle beklenen Q katkısı2.5,eksik faktörle3;Qσ ufukG=Qa ile n1 hedef3,basılıG0 ile2/1;7.17 içindeρ kullandıktan sonra ikinci dışρ,iki-sonuç beklentisini1.5→2.25 yapar. KaynakQσ-CV'nin σ1 ucu7.14kontrol-değişkenli biçimdir;düzSARSA7.4 ile her yolda aynı değildir. Doğru n1 sınırı Expected hedefini verir. İlgili2017/2018AAAI karışıkδσ biçimi dış doğrulama kaynağı olarak açık ayrıldı;kitabın bilgi sayımına ek olgu diye katılmadı. Donmuş V için7.3 kontraksiyonu,gelecekteki uyarlanan tabloyu koşulsuz bağımsız saymaz.

`measurements_ch7_js.json`,bütün16 yeni ön ayarın gerçekten çalıştırılmış ölçümlerini ve24 n/σ/terminal yedekleme örneğini saklar. Bütün210 kaynak hata ve63 saklıδ noktası ayrı Python ile karşılaştırıldı; kaynak için en büyük fark1.11×10⁻¹⁵. CV'nin200 hata noktası veσ0Tree/σ1CV/hata-toplamı özdeşlikleri de denetlendi.

SVG üreticisi kaynak Şekil7.4 rotasını17 yasal hücre hareketi olarak doğrular;bir/on-adımda1/10 pozitif Q değişimini hesaplar. Kaynak pozitif ödülün büyüklüğünü vermez;+1/α.5 bizim ölçeğimizdir. Üç-adımTree'nin7 yaprakπ çarpımı1 eder;indirimli ödül/yaprak hesabı hedef3.048625 verir. Sonraki durum terminal olduğunda A_T veyaρ_T yoktur.

Eski x3 API'nin N/gamma parametreleri varken RMS referansı19durum/γ1 olarak sabitti. Varsayılan çağrı ve scalar dönüş korundu;N5/19,sol0/−1,başlangıç veisteğe bağlı history eklendi. Genelγ için Thomas çözümü,16 ayrı NumPy Bellman sistemiyle karşılaştırılır;γ0 veγ1 kapalı sınırlar ayrıca test edilir. Kaynak Şekil7.2 ilk10 ortalaması,x3 eski varsayılanı son-RMS döndürür. x5/x6 veQ hedefleri için x7 tam starter/model/kontrol/Notebook olarak eklendi.
