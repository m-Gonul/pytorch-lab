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
