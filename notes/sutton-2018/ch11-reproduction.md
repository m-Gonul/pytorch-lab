# Bölüm11: yeniden üretim ve veri kapsamı

Kullanıcı PDF kaynağı: s.257–286/PDF279–308.30 metin ve30 PNG sayfası incelendi.156 kaynak birimi,4 alıştırma,6 numaralı+6 numarasız şekil,29 numaralı denklem,4 dipnot; kod kutusu yok. Ayrı ajan yalnız kaynak ve defteri yeniden okudu;13 bulgu grubu ve takip denetimi kapandı. Üretim işi devredilmedi. Telifli paragraf, piksel veya hazır eğri alınmadı.

Bütün bilimsel Python komutlarında OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 kullanın. NumPy/SciPy ve gerçek CPU PyTorch gerekir.

- `g++ -O3 -std=c++17 measure_ch11_samples.cpp -o /tmp/ch11-samples` ardından `/tmp/ch11-samples > measurements_ch11_samples.json` komutuyla JSON üretir. Dört100000-geçiş davranış yolu, ilk1000geçişte dört değer öğrenicisi, her yolda iki-blok16parametreli Q-learning; ayrıca4096tohum×1000gerçek-geçiş ETD. Tohumlar1,2,1000001,1000002; kuyrukta1…4096. Hiçbir örnek sonuca göre seçilmedi.
- `python measure_ch11_reference.py` normal denklemler, beklenen DP/TDC/GTD2/ETD yörüngeleri, dört-amaç geometrisi, iki alias-model ailesi, rasyonel koşul karşı örneklerini üretir. Çıktı ölçüm JSON dosyasında saklanır. Beklenen güncelleme, adaptif örnek ağırlıklarının tam beklentisi değildir.
- `python verify_ch11.py` ayrı Python geçiş döngüsüyle C++ fixture/ağırlıklarını, QR ile normal denklemleri, sonlu farkla PBE gradyanını, izdüşüm ve RE özdeşliklerini denetler. Sonuç `measurements_ch11_independent.json`.
- `python generate_ch11.py /tmp/ch11-figures` kendi hesaplarından f22…f33 HTML/SVG gruplarını ve `ch11-etd.js` verisini üretir. Sonuncusu `site/js/data/` içine alınır; figürler ders metnine gömülüdür. SVG koordinatları, ağırlıklar, durum ağları ve amaçlar açık modellerden üretilir.
- `node --test test/ch11-numeric.test.mjs`: beklenen bütün ağırlıklar/yardımcılar, dört örnek değer algoritması, dört100000-adımlı Q ağırlığı, dört amacın minima/ölçümleri, A-split/presplit, ham/merkezlenmiş BE limiti, PBE tutarsızlık ve ETD moment kontrolleri.
- `python test/test_modules.py m11-6-dqn`, `python test/test_notebooks.py m11-6-dqn`: x7 zorunlu kaynak programı ve kendi x8 PyTorch çift-örnek çalışması dahil.
- `python check_ch11_browser.py --base http://localhost:8000 --out /tmp/ch11-browser`:24 önayar×1280/390px,8 kart ve12 figür; SVG metin sınırları, KaTeX, JS, kimlik tekrarı ve sayfa taşması. `check_ch11_ab.py` farklı yöntem/amaç/mod sayılarında A/B görünümünü denetler.

Kaynak11.3 eylem özelliklerini belirtmediği için Q-learning deneyinde her eyleme ayrı8ağırlık seçildi; iki blok aynı kaynak başlangıç vektörüyle başlar. Davranış değişmez. Sonlu deney ıraksayan büyümeyi gösterir; bütün başlatmalar için teorem değildir. ETD sayısal koruması1e100;4096koşunun tamamı bütçeyi bitirdi, sansür yok. Tarayıcı bütçe ve gerçek işlenen adımı ayrıca gösterir.

Düzeltmeler:11.6son eylem beklentisi/oranı;11.8üç parametre zamanı;episodic denilen terminalsizBaird;TDE0 genel-şema etiketi;PBE0 içinAw=bkoşulu;SGD/nonlinear genel yakınsama kapsamı;A-presplit4fiziksel/3gözlem;koşullu ödül deterministikken geçişin rastgele kalması;BEhamγ↑1 limiti(−1/3,1/6) ve B0merkezli(−1/2,0) ayrımı;VE/kökVE;dipnot4Amatrisindeki eksik x−γx′;varyansın her öneri dağılımında zorunlu artmaması;Mahadevanadı. Bütünleri kaynak sayfa ve çalıştırılmış normal denklem/fixture hesabına bağlandı.

Yalnız ETD yinelemesini çapraz okumak için Shangtong Zhang eşlikçi dosyası incelendi:
https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter11/counterexample.py

İncelenen Git blob SHA: dec1120c95e5a2b302bcfce06aa27fc0e42fd3bf. Bu bir **blob** kimliğidir, commit kimliği değildir. expected_emphatic_TD sonlu başlangıç vurgu güncellemesi aynı cebirsel yinelemeyi verir. Kaynak kitaba ait resmi kod olduğu iddia edilmez. Kod kopyalanmadı, dosyanın/depodaki diğer içeriğin tamamı içe aktarılmadı; lisans doğrulanmadı, yalnız link atfıdır. Kitabın Mahadevan2014 kaynakça girdisi s.286 “Mahadeval” yazımını doğrular.

Ölçümle değişen taslaklar: varsayılan oran-ürün uzunluğu10yerine3;10durumunda iki tohumda da sıfır örnek gözlenmesi tohum etkisini gizliyordu, uzun nadir-olay durumu ayrıca korundu. Önceden ölçülen4096koşu medyanında iki orta değerin ortalaması kullanıldı. Çizici xlogdesteklemediği için log10ekseni açık dönüşümle çizildi. ETDkitap eğrisiyle yükseklik farkı zorla ölçeklenmedi; sonlu-mbaşlangıcı ve bağımsız eşlikçi yinelemesi açıklandı.
