# Değişiklik günlüğü

## 2026-11-14 · Yeni kaynak: Sutton ve Barto, Bölüm 5 (s.91–117)

M11.4 ve M11.5 eski anlatım,el hesabı,aşamalar,ekler,kartlar ve deneyler korunarak genişletildi. Yeni modül açılmadı. İlk beş bölüm işlendi;bütün kitap tamamlanmadı.

- **Defter:**161 kaynak birimi:159 eklendi,1 genişletildi(doğal21 programı),1 düzeltildi(s.107 yakınsama);vardı/bekliyor/kapsam-dışı0. İlk beş bölüm toplam707 birim. Kaynağın14 numaralı alıştırması ve2 numarasız sorusu,9 görseli ve5 kutulu sözde-kodu karşılandı. Sabit sonlu ufukta ayrı normalize edilmiş PDIS kanıtı kendi öğretim uzantısı olarak ayrı tutuldu.
- **Anlatım ve kartlar:**ilk/her ziyaret,önyükleme bağımlılığı ile istatistiksel bağımlılık ayrımı,Blackjack doğal21/200 durum,Dirichlet tel modeli,MC-Q/GPI/ES,ε-yumuşak iyileştirmenin tam türetimi,kapsama/yol oranları,ordinary/weighted yanlılığı,sonsuz varyans,artımlı ve indirim/karar başına tahmin. 16 yeni çözümlü kart:14 numaralı ve2 numarasız soru. M11.4 toplam18,M11.5 toplam10 kart. Bütün tarihsel atıflar ve2018 araştırma bağlamı korunur.
- **Görseller:**3 hesaplanan SVG grubu,4 yeni deney türü ve5 yerleşim. blackjackmc iki modülde Şekil5.1/5.2;soapdirichlet tel/yüzey/P/h/x/y/izdüşüm;importancemc Şekil5.3/5.4 ve indirim/karar başına karşılaştırma;racetrackmc iki tam kaynak pisti. Kaynak fotoğrafı kullanılmadı;Hersh–Griego/Scientific American kredi bilgisi aktarıldı. Şekil5.2 politika sınır etiketi11 ve Şekil5.3 kaynak0/log1 farkı açıklandı.
- **Kod:**M11.5 x4 Alıştırma5.12 için iki pistin hücreleri,segment fiziği,ortak%10 ivme hatası,çarpışmada aynı bölümde sıfır hızla başlangıca dönme,weighted MC ve yörüngeler. x5 Alıştırma5.14 için Q oranını t+1 ile başlatan indirim duyarlı kontrol;γ<1 kısa önekleri korur. Her iki ek starter/model çözüm/kontrolle notebook'a girdi. M11.4 x1 doğal21 kuralı düzeltildi;eski karşılaştırma çeşidi isteğe bağlı korunur.
- **Hesap:**200 Blackjack durumunun bütün değerleri,Q ve krupiye dağılımları bağımsız NumPy/JS ile eşleşti. Hedef aslı13/krupiye2 için kaynak−0.27726,model−0.277204191825. 500000 gerçek tahmin bölümü RMS Python0.0286822/JS0.0261107;1 milyon ES politika uyumu iki uygulamada199/200. Farklı RNG aynı elleri üretmez. İki pistin19181 durumunun bütün-yasal-eylem optimum değerleri de eşleşti;en büyük fark3.56×10⁻¹⁵.
- **Uzun deneyler:**iki pistte50000 gerçek MC bölümü;sol1049077 adım/modelV−12.112482/öğrenilmişV−14.458549,sağ784925/−8.951270/−11.447987. Bütün başlangıçlardan gürültüsüz öğrenilmiş ve model-optimum yörüngeleri bitirdi. Sonlu MC sonucu kanıtlanmış optimum diye sunulmaz. Şekil5.3:100×10000 gerçek bölüm,son MSE ordinary0.000956878/weighted0.000924055. Şekil5.4:10×100 milyon bütçenin kümülatif kontrol noktaları çokterimli kategori sayımıyla tam dağılımda örneklenir;80 sonrası kuyruk da korunur. Bir milyar yolun tek tek yürütüldüğü iddia edilmez.
- **Kaynak düzeltmesi/notları:**s.107 ordinary IS “yakınsamaz” iddiası yanlış:EX=E|X|=1,EX²=∞;güçlü büyük sayılar yasası yine yakınsamayı sağlar. Durrett Teorem2.4.1 birincil kanıt atfıdır. Tek örnek WIS beklentisi için sıfır oran/payda kuralı ayrılır. Gelecek önem oranları koşulsuz bağımsız sayılmaz;koşullu ortalama1/kule özelliği kullanılır. Genel sabun minimal yüzeyi harmonik yüzeyle özdeş değildir;küçük-eğim yaklaşımı açıklanır. MC ES keşif başlangıçlarını korur;2018 açık sorun ifadeleri güncel literatür iddiası değildir.
- **Ölçümle düzeltilen taslak:**sonlu MC ES ve yarış politikasını tam optimum adlandırma kaldırıldı;gerçek model farkları yazıldı. Kaynak−0.27726 eksisi görüntüden doğrulandı. Hareket son hücre yerine segmentle denetlenir;ilk kesişim ve köşe teması açık uygulama seçimi olarak işaretlendi. Başlangıç çizgisinde hedef sürücünün iki son hızı0 yapan eylemleri elemesi tüm-yasal-eylem modeliyle doğrulandı;iki haritanın optimumV değerlerini değiştirmedi. Kaynak renkleriyle özgün çizimin turuncu başlangıcı ayrıldı.
- **İkinci okuma:**bağımsız ajan yalnız PDF113–140 ve defteri okuyup27 metinli sayfa/1 boş sayfa ve12 kaynak görüntüsünü denetledi. İki takipte bütün somut bulgular kapandı;161 birim ve öğe sayıları doğrulandı.8 sözlük tanımı eklendi.
- **Doğrulama:**tam Node koşusunda754/755 geçti;tek hata,çalışma sürerken değiştirilen x4 açıklaması nedeniyle eskimiş notebook idi. Notebook'lar yeniden üretildi;son site denetimi7/7 ve yeni bağımsız sayısal testler4/4 geçti. Değişen iki modül ve iki notebook gerçek CPU PyTorch ile2/2 temiz. 1280/390 px için44 yeni ön ayar çalıştırıldı;taşma,KaTeX/JS hatası veya yinelenen kimlik yok. Eksen yazısı kırpması ve eski #q/arama-alanı kimlik çakışması düzeltildi. Önceden içeriğe gidemeyen #q çapası #q-deger oldu;başka iç bağlantısı yoktu. Yayın CI ayrıca izlenir.
- **Kalan:**Bölüm6–17,ön/arka materyal,kısım ayırıcıları,tam kaynakça/dizin. Küresel complete/unit_inventory_complete/second_read_complete false kalır. Önceki Bölüm4 CI/Pages başarılı ve canlı içerik doğrulandı;bu bölümün push işlemi tek başına yayın kabul edilmez.

## 2026-11-13 · Yeni kaynak: Sutton ve Barto, Bölüm 4 (s. 73–90)

M11.3 mevcut içeriği korunarak genişletildi; M11.2 içindeki aynı V*/Q* denklemlerine kesin bağlantı verildi. İlk dört bölüm işlendi, bütün kitap tamamlanmadı. Yeni modül yok.

- **Defter:** 102 kaynak birimi:99 eklendi,2 vardı,1 genişletildi;düzeltildi/bekliyor/kapsam-dışı0. İlk dört bölüm toplam546 birim. Poisson kuyruk/matris türetimi kaynak sayfada verilmediği için öğretim eki olarak ayrı tutuldu,102 sayısına katılmadı. Kaynakta10 numaralı alıştırma,1 numarasız soru,7 görsel ve3 kutulu sözde-kod tamamlandı.
- **Anlatım:** Bellman eşitliğini atamaya çevirme;beklenen güncelleme;PE için ortak p/Jacobi/Gauss–Seidel/durma;getiri açılımlı iyileştirme kanıtı;stokastik bağlar;PI/VI sözde-kodlarının bütün aşamaları;ε-yumuşak kısıt;Jack/kumarbaz;asenkron sıra ve ziyaret koşulları;GPI rekabet/işbirliği;verim,boyut laneti,özyükleme ve bütün tarih atıfları.
- **Kartlar:** 10 yeni kartla toplam 16. Kaynak4.4 sorusu aynı mevcut p5 kartıyla birleştirildi;eski metin/seçenekler korundu,köken ve tam döngü düzeltmesi eklendi. 4.1–4.3→p7–p9;4.5–4.10→p10–p15;numarasız bütün optimal aile sorusu→p16. Bu yüzden kaynak11 sorusu için11 yeni tekrar kartı üretilmedi.
- **Görseller:** 3 yeni deney ve2 SVG grubu. gridpefig/l3:Şekil 4.1 k0/1/2/3/10/limit için bütün sayılar/oklar ve kurulum ızgarası. jackrental/l4:Şekil 4.2 π0…π4 ve son3B değer yüzeyi;4.7 varyantı. gamblerfig/l5:Şekil 4.3 1/2/3/32 tarama,sonV ve politika;4.9 için0.25/0.55/gevşekθ. PI zinciri ve iki GPI şeması hesaplanan özgün SVG;ikincisinde gerçek izdüşüm okları kullanıldı.
- **Programlama:** x4 tam .md/.py/.sol.py/.check.py ile4.7. Özgün ve değişmiş problem441 durumda;özgün V(0,0)=421.414063,V(20,20)=636.989607;değişmiş429.946305/603.536701. İkisinde5 politika değerlendirmesi,son sabitlik denetimi dahil. İlk1→2 araba ücretsiz,ters yön ücretli,taşıma sonrası >10 için şube başına 4. x3 gösterimi/arayüzü korunup4.9 ikiolasılık/üçeşik/grafik yönergesi eklendi.
- **Bağımsız sayısal denetim:** NumPy ve JS bütün882 Jack değeri,iki modelin bütün politika dizileri,3 kumarbaz V/politika sonucu ve1/2/3/32 izleri,5 ızgara taraması üzerinden eşleşti. Ek durum15 her iki4.2 dinamiğinde−20;qπ(11,aşağı)=−1,qπ(7,aşağı)=−15. p_h=0.55 çözümünün bir-birim bahis kapalı formuna en büyük farkı8.99×10⁻¹¹. 0.25/0.55 için θ10⁻⁶→10⁻¹² en büyükV farkları0/0.000093520231;9ondalıklı bağ kırmayla politika değişimi0. B*(26)={1,24,26},B*(51)={1,49} sayısal doğrulandı. Mevcut x3 eşzamanlı kodu da 0.25/0.55 ve üç eşikte doğrudan çalıştırıldı; en büyük V farkları 1.74812×10⁻⁷ / 0.000189500726, politika değişimi 0.
- **Kaynak notları:** 0 bahis veγ1,sonlanmayan öz döngüye yol açabilir;pozitif bahis veya proper politika gereksinimi açıklandı. V100=1 sınır değeriyle hedef+1 ödülü birlikte çift sayılmaz. Kaynak−1.75→−1.7 yarımda sıfıra doğru yuvarlama seçimiyle yeniden üretildi. Polinom/LPyaklaşık100kat/milyonlarca durum iddiaları koşulları ve2018 bağlamıyla yazıldı. Tam Poisson kuyruğu ve bağımsız sayımların standart problem yorumu açık belirtildi;kaynakta olmayan türetim ona atfedilmedi.
- **İkinci okuma:** bağımsız ajan yalnız PDF95–112 ve defteri okudu;18sayfa/9görüntü denetlendi. Numarasız aile sorusu,VI keyfi başlangıç/deterministik çıktı,GPI her iki süreçte kapsam,Werbos/Minsky tarih nüansları,Şekil 4.2 etiketleri ve sayfa aralıkları tamamlandı. Takip okumadaki iki kısalma da geri kondu.
- **Doğrulama:**745/745 Node testi;M11.3 gerçek CPU PyTorch modül1/1 ve notebook1/1 temiz. Son build/notebook üretimi ve site7/7. 1280/390 px bütün yeni deney ön ayarları/görselleri incelendi;sayfa taşması,KaTeX hatası veya yinelenen kimlik yok. Eski geniş karşılaştırma tablosu kendi alanında yatay kaydırılır;bu mevcut davranış korunmuştur. Altı sözlük tanımı eklendi.
- **Kalan:** Bölüm 5–17,ön/arka materyal,kısım ayırıcıları ve tam kaynakça/dizin henüz tamamlanmadı. Küresel complete/unit_inventory_complete/second_read_complete false kalır. Önceki bölümün CI yayını ayrıca izlenir;push tek başına yayın sayılmaz.


## 2026-11-13 · Yeni kaynak: Sutton ve Barto, Bölüm 3 (s. 47–71)

M11.2 mevcut anlatımı, el hesabı, aşamaları, ekleri, kartları ve deneyleri korunarak genişletildi. Kitabın ilk üç bölümü işlendi; bütün kitap tamamlanmadı. Yeni modül açılmadı.

- **Defter:** Bölüm 3 için 179 birim: 178 eklendi, 1 vardı; genişletildi / düzeltildi / bekliyor / kapsam-dışı 0. İlk üç bölüm toplamı 444 birim. Sınırlı ödül ve γ<1 getiri sınırı aynı derinlikte zaten vardı. Diğer yeni kaynak ayrıntıları kesin çapalara bağlandı. Önceki bölümlerde eskimiş eşleme durumu da doğrulanan birim hedefleriyle güncellendi.
- **Anlatım:** ajan–ortam sınırı ve üç sinyal, ödül indeksleri eşlemesi, ortak p ve üç marjinali, Markov temsil yeterliliği, biyoreaktör/al–yerleştir/robot, ödül hipotezi ve dipnotlar, bölüm/devam eden görev, soğurucu terminal ve ortak getiri, stokastik politika, V/Q beklenti ve en iyilik denklemleri, golf, ızgara, robot, tam/eksik model bilgisi, hesap/bellek ve bütün tarih atıfları.
- **Sorular:** 3.1–3.29 ayrı çözümlü p7–p35 kartları; eski 6 kart korundu, toplam 35. Hesap, kanıt, modelleme, karşı örnek, yorumlama ve hata bulma türleri. Sayısal kartların calc ifadeleri çalıştırıldı; 3.8 getirileri (2,6,8,4,2,0), 3.9 (G1,G0)=(70,65), 3.14 yuvarlak girdilerle 0.675, 3.24 tam değer 24.419428 / üç ondalık 24.419.
- **Görseller:** 5 numaralı ve 8 numarasız kaynak görseli karşılandı; 6 yeni SVG grubu ve 2 deney eklendi. Ajan çevrimi, robotun beş eylemi/yedi dalı, çubuk ve soğurucu durum, golf, altı backup paneli ve üç durumlu MDP; numaralar sayfada 3–8 olarak devam eder. Kaynak Şekil 3.2 ve 3.5, suttongrid/l4 içinde ayrı γ=0.9 ön ayarlarıdır; 25 değer ve bütün eşit en iyi oklar korunur. Robotun 10 satırlı tablosu üç sıfır olasılıklı satırı da içerir.
- **Hesap:** NumPy ile 25×25 rastgele politika çözümü ve bağımsız değer yinelemesi, JS çözücüyle bütün 50 değer/tüm oklar üzerinden karşılaştırıldı. A rastgele 8.789292, en iyi 24.419428; orta hücre 0.673113. Altı robot politikası bağımsız 2×2 çözüldü; seçilmiş sayısal örnekte Vh=8.474576, Vl=7.627119. Özgün golf modeli tee için 6 putt veya 2 drive+putt verir. Betik ve ölçümler notes/sutton-2018 içinde.
- **Kaynak notları:** 3.4 içinde koşullu ortalamalar ortak ödül dağılımını tekil belirlemez; genel ρ çözümü ve açık belirlenimci-ödül ek varsayımı verildi. Bellman tekliği γ<1 için yazıldı; γ=1 sıfır ödüllü döngü karşı örneği ve sonlanma koşulu açıklandı. Golfün eksik geometri/olasılık verilerinden tekil sayısal saha uydurulmadı; özgün model açık etiketlendi. Çubuk cezasında −γᴷ ile K−1 üsleri arasındaki sayım farkı, tek ceza ve bütün devam eden getiri ayrımıyla açıklandı. Yeni bir kesin kaynak hatası iddiası yok.
- **İkinci okuma:** ayrı ajan yalnız PDF69–93 ve defteri okuyarak 25 metin sayfasını, 12 şekilli görüntüyü denetledi; üretimi yapmadı. İşaret alt sorusu, π koşulları, Markov/Kısım II, tam bilgi, tarih nüansları ve kaynak sayfaları tamamlandı. 3.5 s.55, 3.20 s.63; 3.15 s.60–61, MC s.58–59 aralıkları düzeltildi. Kaynak qπ şemasındaki 2×3 ve q* şemasındaki 2×2 ayrımı korundu.
- **Görsel düzeltmeler:** tanımsız arka plan değişkeni açık düğümleri siyaha boyuyordu; site rengiyle düzeltildi. Robot çubukları ve kenar etiketleri için x aralığı genişletildi, y eksenine sıfır dahil edildi. Golfte aynı green alanı içinde driver tek-vuruş bölgesi daha küçük çizildi. Kuvvet oku/açı etiketinin kaynak çizime ek açıklama olduğu belirtildi. Birden fazla alt yolu olan SVG çizgilerinde yalnız son ok başı görünüyordu; A/B, çevrim ve üç durum dönüş yolları ayrı çizilerek bütün ok başları gösterildi.
- **Doğrulama:** 740/740 Node testi. Son grafik düzenlemesinden sonra ilgili 4 deney testi tekrar geçti. M11.2 gerçek CPU PyTorch modül ve notebook testleri 1/1 temiz. 1280/390 px ekran görüntüleri, matematik, eşsiz kimlikler ve taşma kontrol edildi. 12 sözlük tanımı eklendi. İlk iki bölümün CI ve Pages yayını başarıyla tamamlandı, canlı kaynak işaretleri doğrulandı.
- **Kalan:** Bölüm 4–17; ön/arka materyal, kısım ayırıcıları, tam kaynakça ve dizin henüz tümüyle işlenmedi. Kitabın küresel tamlık ve ikinci okuma bayrakları false kalır. İşlenmemiş hazır hedefler yanlış anlamda bekliyor ile kapatılmadı.


## 2026-11-13 · Yeni kaynak: Sutton ve Barto, Bölüm 2 (s. 25–45)

M11.7 mevcut içeriği korunarak genişletildi. Kitabın ilk iki bölümü işlendi; bütün kitap henüz tamamlanmadı. Yeni modül açılmadı.

- **Defter:** Bölüm 2 için 106 birim: 104 eklendi, 1 genişletildi, 1 düzeltildi; vardı / bekliyor / kapsam-dışı 0. İlk iki bölüm toplamı 265 birim. Kaynak s.25–45 ve altı şekil bağımsız ikinci okumayla denetlendi; 18 ayrıntı kaydı, Denklem 2.13 eşlemesi ve Kumar'ın Bayesçi/Bayesçi olmayan ayrımı düzeltildi.
- **Anlatım:** değerlendirici/öğretici geri bildirim, kaynak gösterimi eşlemesi, örnek ortalaması ve sabit bellek türetimi, üstel ve değişken-adım ağırlıkları, yakınsama koşulları, iyimser başlangıç, UCB deney koşulları, softmax türevi ve gradyan bandit, bağlam ipucu, Bayesçi bilgi durumu ve kaynak tarihindeki bütün atıf–işlev ilişkileri.
- **Sorular:** 2.1–2.11, bütün değer/koşul ve alt sorularıyla p7–p17. Mevcut 6 kart korundu; toplam 17. 2.5 için x2, ödül eğrisini de döndüren isteğe bağlı arayüzle genişletildi; varsayılan eski dönüş korundu. 2.11 için tam .md/.py/.sol.py/.check.py dörtlüsüyle x4 eklendi. Sayısal 2.1, 2.2, 2.3 ve 2.10; ağırlık kanıtları ve taban düzeltmesi çalıştırılarak doğrulandı.
- **Altı şekil:** 2.1 için kendi Gauss çekilişimizden hesaplanan SVG; 2.2–2.6 için suttonbandits/l4 içinde ayrı kaynak ön ayarları. Beş grafik senaryosu gerçek örneklerden yeniden hesaplanır, 2000 görev × 1000 adım ve kaynak parametreleri korunur. 11. adımı kaybetmemek için ilk 30 nokta saklanır. Kendi tohumumuz kaynak eğrisinin rastgele gürültüsünü aynen kopyalamaz.
- **Uzun deneyler:** 2.5, 2000 görev × 10000 adım, tohum 1. Son 1000 adımda örnek ortalaması/sabit α=0.1 için ortalama ödül 1.087398/1.343691, doğru kol oranı %46.46005/%76.21655. 2.11, 8 tekrar × 200000 adım, 32 yöntem/parametre ayarı, tohum 1; ölçüm yalnız 100001–200000 aralığı. Bu koşuda en yüksek ortalama sabit α=0.1, ε=1/128 için 5.787817. Sekiz tekrar bir kaynak sabiti değil açık hesap bütçesidir; sonuç bütün problemler için üstünlük iddiası değildir. Ham ölçümler ve yeniden üretim betiği notes/sutton-2018 altında.
- **Kaynak düzeltmesi:** s.37–40, güncel ödül dahil örnek ortalaması tabanı ile tam gradyan eşitliğinde (1−1/t) çarpanı eksiktir. Cebir ve bağımsız üç kollu sonlu farkla doğrulandı. Kaynak eğrilerini üreten kod güncel ödül dahil tabanı kullanır; değişen etkin adım açıklanır. Gauss ödüllerine sınırlı ödül Hoeffding kanıtının doğrudan taşınamayacağı kaynak notunda ayrıldı.
- **Ölçümle düzeltilen taslak:** PDF metni 2.2 içindeki −1 ve −2 işaretlerini atlıyordu; sayfa görüntüsünden geri alındı. Kaynak 2.13 numarası yanlışlıkla performans toplamına eşlenmişti; tam gradyan adımı olarak düzeltildi. Gürültüsüz ödül kontrolünde kayan nokta toplamı 1.0000000000000007 verdi; sınır kontrolüne 10⁻¹² sayısal tolerans eklendi. Yoğunluk görselinin aralığı kuyrukları kesmeyecek biçimde düzenlendi. Kumar'ın kategori ayrımı sıklıkçı diye daraltılmadı.
- **Mevcut x2 düzeltmesi:** bütün q* değerleri başlangıçta eşitken argmax yalnız ilk kolu optimal sayıyordu. Gerçek değer eşitliğiyle bütün en iyi kollar artık doğru sayılıyor; ilk adımın %100 olması bağımsız kontrolde doğrulanıyor. Bu değişiklik son 1000 adım ölçümlerini değiştirmiyor.
- **Sözlük:** 16 tanım eklendi; M5.5 ve M11.7 içinde ilk kullanım ipuçları bağlandı.
- **Doğrulama:** 736/736 Node testi; son eksen düzenlemesinden sonra ilgili 3 lab testi tekrar geçti. M11.7 gerçek CPU PyTorch modül testi ve güncel notebook testi 1/1 temiz. 265 birim hedefi ve bölüm tür sayıları doğrulandı; 1280/390 px görüntüler incelendi, taşma ve tarayıcı hatası yok. İlk notebook kontrolü eski sınır kontrolünü içerdiği için düşmüştü; notebook yeniden üretilip uçtan uca test edildi.
- **Kalan:** Bölüm 3–17, ön sayfalar, tam kaynakça ve dizin henüz işlenmedi. Kaynağın bütünü için tamlık veya yayın tamamlandı iddiası yok.


## 2026-11-13 · Yeni kaynak: Sutton ve Barto, Bölüm 1 (s. 1–22)

Kitabın aktarımı başladı; bütün kaynak henüz tamamlanmadı. M5.5 mevcut içeriği korunarak genişletildi. Yeni modül açılmadı.

- **Envanter:** 548 PDF sayfası, 17 bölüm; 140 numaralı alıştırma, 104 numaralı şekil, 38 kutulu sözde-kod listesi, 1 numaralı tablo. Bunlar etiketli öğe sayılarıdır; numarasız çizimler, tablolar ve metin içi algoritmalar bütün kitapta henüz taranmadı. Kaynak dosya özeti ve sayfa bazında metin karmaları `content/coverage/sutton.yaml` içinde kayıtlıdır.
- **Bölüm 1 defteri:** 159 birim: 158 eklendi, 1 düzeltildi; vardı / genişletildi / bekliyor / kapsam-dışı durumlarının her biri 0. Bu sayılar yalnız işlenen bölümün birimleridir. Diğer bölümlerdeki eksikler, hazır hedeflerde yanlış biçimde “bekliyor” olarak kapatılmadı.
- **Anlatım:** problem/yöntem/alan ayrımı; beş kaynak örneği; politika, ödül, değer, model; temsil ve evrimsel politika araması; XOX başlangıç değerleri, hamle sonrası TD ve keşif; üç tarihsel hat ve Bölüm 1 kaynak notları. 9 yeni çapa.
- **Sorular:** Alıştırma 1.1–1.5, bütün alt soruları ve adım adım çözümleriyle p6–p10 kartlarına eklendi. Var olan 5 pratik kartı korundu; toplam 10. Kaynak soruları açık uçlu olduğu için cevap türleri korunmuştur.
- **Görseller:** numarasız XOX tahtası ve Şekil 1.1, iki özgün SVG ile karşılandı. Tahtadaki kazanan çapraz ile örnek yörüngenin yasallığı tam oyun ağacında hesaplandı. Açgözlü c/g seçimi, daha iyi ama seçilmeyen e*, keşifte güncelleme yapılmaması, c→a ve g→e öğrenme okları korundu. Kaynak görüntüler yayımlanmadı.
- **Deney:** explorevalue (l3), keşifli davranış ile açgözlü hedefi tam olasılık hesabıyla ayırır; dört ön ayar, tahmin sorusu ve sınırlar eklendi. ε = 0.3, güvenli başarı 0.8 için riskli dal 0.7; kök başarıları 0.73 ve 0.77. ε = 0 durumunda ikisi de 1; güvenli başarı 0.4 durumunda ikisi de 0.61. Bu bir XOX eğitim benzetimi değildir.
- **Kaynak düzeltmesi:** s. 11 içindeki sinir ağı atfı §9.6 yerine §9.7 olmalıdır; aynı PDF içindeki s. 222–223 başlıklarıyla doğrulandı. Lisans, yüklenen PDF 4 üzerinden CC BY-NC-ND 2.0 olarak doğrulandı; kullanım link olarak kaldı.
- **Ölçümle düzeltilen taslak:** masaüstü/mobil görüntü denetiminde yeni SVG düğüm yazıları küçük, kesikli dallar ve tahta çizgileri soluktu; yazı boyutu ve çizgi karşıtlığı artırıldı. Bağımsız ikinci okumada 26 ayrıntı kaydı ve kaynak soruların alt koşulları defterde genişletildi.
- **Doğrulama:** 733/733 Node testi, değişen M5.5 için gerçek CPU PyTorch ile modül testi; build ve notebook güncellik denetimi geçti. 1280 ve 390 px tarayıcı denetiminde yatay taşma, taşan satır içi matematik, yinelenen HTML kimliği veya sayfa hatası yok. Sayısal deney ön ayarları bağımsız yaprak olasılık toplamıyla doğrulandı. Son metin/kapsam değişikliklerinden sonra 7 site testi tekrar geçti.
- **Kalan:** Bölüm 2–17, ön sayfalar ve kitabın tam kaynakça/dizin içeriği işlenmedi. Bütün kitap için ikinci okuma veya sıfır kayıp tamamlık iddiası yok. Eski Toygar slaytları bu kaynak dosyaları arasında olmadığından yeniden görsel denetim yapılmadı.


## 2026-11 · M1.3 derinleştirme: elastic net ve erken durdurma türetimleri, üç yeni deney, CV ile λ seçimi

M1.3 (regularization) ekleme odaklı genişletildi; hiçbir mevcut içerik kaldırılmadı.

- **Teorik eklentiler (lesson.md):** elastic net'in tek koordinat türetimi ($w^\star=S(z,\lambda_1)/(1+\lambda_2)$, $\lambda_1=\alpha\lambda$, $\lambda_2=(1-\alpha)\lambda$: önce L1 eşiği, sonra L2'nin paydaya girdiği oransal büzülme); erken durdurma–ridge eşdeğerliği türetimi (ηη sabit adımda k adımdan sonra etkin ceza $\lambda\approx 1/(k\eta)$); standartlaştırmanın önemine 10⁴ oranlı sayısal örnek (aynı katsayı, birimsiz öznitelikte ceza birimli olandan 10⁴ kat küçük).
- **Mevcut deneyin genişletmesi:** l1l2geom'a (L1/L2 topu) p = 2 elastic net seçeneği eklendi: ISTA ile çözüm, bisection ile λ(t) (yarıçapın kesin karşılanması) ve $\alpha\|w\|_1+(1-\alpha)\|w\|^2/2=t$ top şekli; α kaydırıcısıyla köşelerden yuvarlağa geçiş canlı izlenir.
- **Üç yeni deney:** l4 shrinklab (tek koordinatta lasso $S(z,\lambda)$, ridge $z/(1+\lambda)$ ve elastic net eğrileri; sürüklenebilir z, gölgeli sıfırlama bölgesi; selfTest'li custom lab), l5 svdlab (ridge'in SVD bakışı: yön-yön çarpanlar $s_j^2/(s_j^2+n\lambda)$, df(λ) etkili serbestlik ve koşul sayısı iyileşmesi tablosu), l6 cvlab (k-fold CV ile λ seçimi ve 1-SE kuralı; merkezleme her fold'un yalnız eğitim kısmından — sızıntı yok; ridge yolu, fold'un merkezlenmiş Gram matrisinin özdeğer ayrışımıyla $w(\lambda)=V\,\text{diag}(1/(\mu+\lambda))\,V^\top c$ biçiminde tüm λ'lar için birden hesaplanır).
- **Ek alıştırma E1 (x1):** elastic net'i koordinat inişiyle kodlama — λ_max ≈ 2.27'ye inen path, ilişkili grup için gruplama etkisi (özdeş katsayılar, fark < 10⁻⁶), ilişkisiz grupta lasso'nun split davranışı; çözüm check'ten geçer, boş iskelet geçmez.
- **Sorular:** +2 quiz (5'e; k-fold CV'de standartlaştırma sızıntısı, elastic net tek koordinat çözümü), +2 pratik (p7 elastic net tek koordinat hesabı, p8 1-SE kuralı ile λ seçimi), +1 what-if (η'yi yarılayıp k'yi ikiye katlamanın erken durdurma cezasına etkisi), +1 yanlılık (CV'de tüm veride standartlaştırmak).
- **Sayısal doğrulama:** üç yeni deneyin aritmetiği referans script'lerle doğrulandı (cvlab: Gram hilesi doğrudan çözümle her λ'da < 10⁻⁹ çakışıyor; shrinklab: selfTest; svdlab: df ve çarpanlar kapalı formla); 705 deney testinin tümü geçti, modül testi temiz, M1.3 notebook'u yenilendi ve uçtan uca geçti.

## 2026-11 · M1.2 derinleştirme: kararlılık bölgesi türetimi, üç yeni deney, görsel genişletmeler

M1.2 (gradient descent varyantları) ekleme odaklı genişletildi; hiçbir mevcut içerik kaldırılmadı.

- **Teorik eklentiler (lesson.md):** ağır toptaki 2×2 blok $M_\lambda$'nın özdeğer analiziyle kararlılık bölgesi kanıtı (Jüry koşulu → $\eta\lambda<2(1+\beta)$, spiral bandı $\eta\lambda=(1\mp\sqrt\beta)^2$, mercek ve ucu $\beta^*,\eta^*$'ın geometrik türetimi); κ → adım sayısı tablosu (GD vs momentum, 10⁻³'e); Nesterov'un erken frenleyen $1-\eta\beta\lambda$ terimine kısa türetim; EMA-biçimi $v\leftarrow\beta v+(1-\beta)g$ ile ağır top arasındaki $\eta(1-\beta)$ eşdeğerliği kutusu; Polyak–Ruppert ortalaması ayrıntı kutusu; Robbins–Monro takvim tablosu (sabit, 1/t, 1/√t, t^−α, basamaklı); karıştırma tanım kutusuna sıralı veri uyarısı.
- **Mevcut deneylerin genişletmesi:** gd1d'e kobweb diyagramı (y = x karşı y = (1−ηa)·x) ve (η, a) rejim haritası + $|1-\eta a|^T$ ölçümü; sgdnoise'a adım bazlı kayıp grafiği, birinci mertebe teorik taban $(\eta/4)\operatorname{tr}\Sigma_g$ referans çizgisi (sayısal doğrulandı) ve "x'e göre sıralı" veri seçeneği (karıştırmanın zararının canlı gösterimi); optim2d'e karşılaştırma katmanı (SGD + Momentum + Nesterov aynı η, β ve gürültü akışıyla üst üste), dik-yön zaman serisi ve (η, β) kararlılık haritası + blok spektral yarıçap ölçümü.
- **Üç yeni deney:** l4 hbmap (ağır topun (η, β) kararlılık haritası: ıraksama, spiral bandı ve mercek eğrileri, yön ayrık $\rho^t$ sönüm serileri), l5 sweepmap (Aşama E'nin 48'lik PyTorch sweep ızgarasının tarayıcı haritası: son kayıp ısı haritası + H = 2XᵀX/n'in gerçek özdeğerleriyle kararlılık eğrisi ve mercek ucu), l6 gddrag (sürüklenebilir x₀ ile parabol üzerinde elle adım; selfTest'li custom lab).
- **Sorular ve kaynaklar:** +3 pratik (p8 etkin adım η/(1−β), p9 κ = 64'te adım sayısı ~√κ kazancı, p10 ηₜ = c/t^0.6'nın Robbins–Monro koşulları), +2 quiz (β = 0.99'un tehlikesi; EMA/ağır top biçim farkı), +1 what-if (gürültülü gradyanda β seçimi); kaynaklara Polyak (1964), Nesterov (1983), Polyak & Juditsky (1992), Sutskever vd. (2013) eklendi.
- **Sayısal doğrulama:** mercek ucu formüllerinin (β*, η*) sayısal optimumla birebir çakıştığı; η* = (2/(√λmax+√λmin))²'nin iki sınır eğrisinin kesişimi olduğunu; 1-B gürültü tabanı E[L−L*] = ηs²/(4B) ≈ (η/4)·tr Σg yaklaşımının Monte Carlo ile doğrulandığı (0.0260 ölçüm, 0.0263 teori).

## 2026-11 (3. tur, devam) · M1.1 derinleştirme: türetimler, şekiller, yeni deneyler ve ek alıştırma

M1.1 (doğrusal regresyon) içeriği ölçümle doğrulanarak genişletildi. Ekleme odaklı çalışıldı; hiçbir mevcut içerik kaldırılmadı.

- **Teorik eklentiler (lesson.md):** MSE = Gauss gürültüsü altında en çok olabilirlik türetimi; öznitelik genişletme bölümü (artıklarda kalan eksik terim); hunimsi yayılımda ağırlıklı en küçük kareler ayrıntı kutusu; kararlılık türetimine κ → adım sayısı tablosu (κ = 10 → 69, κ = 100 → 691, κ = 10⁴ → 69 078 adım); mini-batch gradyan varyansının Var(gᵢ)/B ile ölçeklenmesi.
- **Yeni şekiller:** Anscombe dörlüsü (değerlendirme bölümüne) ve kayıp yüzeyi konturu + GD yolu (özdeğerler λ_max = 22.58, λ_min = 1.42, η = 0.07 betikle hesaplandı). linregfit ısı haritasına özvektör yön çizgileri eklendi (ekran-uzayında doğru ölçekleme).
- **Karesel gerçek model düzeltmesi:** "saf 0.8x² − 1" hipotezi ölçümle çürütüldü (doğrusal fit'te R² ≈ 0.03); gerçek model y = 3x + 0.8x² − 1 + ε olarak güncellendi (doğrusal fit'te R² ≈ 0.91 ve U deseni). U imzasının merkezlenmiş x²'ye uydurulması gerektiği (ham x² katsayıyı düşük gösterir) anlatıma işlendi.
- **İki yeni deney:** l3 dragfit (noktaları sürükle: kapalı form veriye tepki verir, doğru (x̄, ȳ)'den geçmeye zorlanır; selfTest'li custom lab) ve l4 linresid (artık tanılama: doğrusal/karesel/huni senaryoları, (x²−x̄²) imzası, R²'yi izleme).
- **Ek alıştırma E1 (x1):** polinom öznitelik genişletmeyi kodlama — Vandermonde φ(x, deg), lstsq ile fit_poly, rmse, artık ve merkezlenmiş x² imzası; U desenli veride imza ≈ −0.87 → ~0, RMSE 1.13 → 0.30, R² (doğrusal) ≈ 0.91 ama biçim yanlış.
- **Sorular ve pratikler:** +2 quiz (η = 2/λ_max sınırı; (n,1) − (n,) broadcasting tuzağı), +2 pratik (p8 özdeğerlerden η* = 2/(λ_max+λ_min), p9 "R² 0.9 ama U deseni" yorumlaması), +1 what-if (η tam 2/λ_max), b–e aşamalarına ipuçları; kaynaklara Anscombe (1973) eklendi.
- **Araç zinciri:** notebook üreticisi kaynak metinleri satır sonu bağımsız okuyor (Windows checkout'unda CRLF kaynaklar notebook'lara \r\n gömüyordu; --check Linux'ta bunu yanlış bayatlık sayardı). M1.1 notebook'u yenilendi (E1 dahil) ve uçtan uca geçti.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 11

Faz 11'in (pekiştirmeli öğrenme) 12 modülünün hepsine tam konu anlatımı yazıldı. Kaynaklar şunlardır:

- YAP 446/BİL 562 Reinforcement Learning dersinin slayt desteleri (lect_2–lect_13);
- DQN makalesi (Mnih ve ark. 2013);
- AlphaGo makaleleri (Silver ve ark. 2016, 2017);
- güven bölgesi modülü için TRPO, PPO ve GAE makaleleri (Schulman ve ark. 2015, 2016, 2017), Kakade (2001) ve Nocedal & Wright (Bölüm 4).

Her deste sayfa sayfa işlendi. Slayt metni kopyalanmadı, türetimler eklendi ve slaytlardaki hatalar "Kaynak düzeltmesi" kutularında gerekçeleriyle düzeltildi. Deneylerin hepsi tarayıcıda çalışan küçük ölçekli simülasyonlardır. Not metinleri ve ön ayar açıklamaları ölçülen sonuçlara göre yazıldı.

| modül | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|
| M11.1 Markov zincirleri | 3 | markovlab, ar1lab | AR(1) durağan varyansı, n adımlık geçiş = matris kuvveti, iki durumlu zincirin kapalı çözümü, temel matris, sayım oranlarının en çok olabilirliği |
| M11.2 MDP ve değer yinelemesi | 2 | valueiter, retaildp, vibounds | indirimli toplamın sonluluğu, iki büzülme lemması, sonlu yineleme sınırı, 2γε/(1 − γ) politika sınırı |
| M11.3 politika yinelemesi | 2 | pimpi, gridpe | politika iyileştirme teoremi, sonlu adımda yakınsama, Bellman artığıyla sınırlar |
| M11.4 Monte Carlo ve TD | 2 | mctd, batchmctd | artımlı ortalama, Robbins–Monro koşulları, ileri ve geri bakış (uygunluk izleri) |
| M11.5 SARSA ve Q-öğrenme | 2 | cliffwalk, maxbias | maksimizasyon yanlılığı ve çift Q |
| M11.6 fonksiyon yaklaşımı ve DQN | 1 | baird, replaylab | yarı-gradyanın neden gradyan olmadığı, ölümcül üçlü |
| M11.7 çok kollu haydutlar | 2 | bandit, ucbbound, betapost | ε-açgözlünün doğrusal pişmanlığı, iyimser stratejinin yakınsaması |
| M11.8 bağlamsal haydutlar | 1 | linucb, ope | Bayesçi doğrusal regresyon, Sherman–Morrison, IPS'nin yansızlığı ve DR'nin iki kez sağlamlığı |
| M11.9 politika gradyanı | 1 | reinforce, pgvar, dpglqr | politika gradyanı teoreminin kanıtı, REINFORCE kestiricisi, taban çizgisinin yansızlığı |
| M11.10 modelli RL ve planlama | 2 | dynamaze, mctsttt | sayım kestiriminin en çok olabilirliği, UCT'nin haydut azaltımı |
| M11.11 AlphaGo'dan AlphaZero'ya | 2 | rootnoise, elogate, aztab | aramanın politika iyileştirme operatörü olması, Elo'nun lojistik regresyon yorumu |
| M11.12 güven bölgesi, TRPO, PPO, GAE | 2 | trustregion, ppoclip, gae | Moré–Sorensen koşulları, KL ≈ ½dᵀFd, doğal gradyan, performans farkı lemması, GAE teleskopu |

- **Faz 11'de 29 deney (26'sı yeni; valueiter, bandit ve reinforce önceden vardı) ve 22 şekil.** Şekillerdeki bütün sayılar betikle hesaplandı.
- **Kaynak düzeltmeleri** (her biri ilgili modülde kutu içinde):
  - M11.1: "durağan Markov süreci" ile geniş anlamda durağanlığın karıştırılması; hesap karmaşıklığı; "bu zincirde herkes satın alır" sonucu.
  - M11.2: "eylemler belirlenimci" varsayımı; ara terimlerdeki yazım hataları; Teorem 2'nin son adımı.
  - M11.3: yakınsama hızları ve değiştirilmiş politika yinelemesinin sınırları.
  - M11.5: $Q^\pi$'nin tanımı ve Q-öğrenmenin yakınsama koşulları.
  - M11.6: doğrusal Q-öğrenme her zaman yakınsamaz; DQN'nin 2013 ve 2015 sürümleri.
  - M11.7: Beta dağılımının yorumu.
  - M11.8: öngörü dağılımı.
  - M11.9: amacın koşullu yazılması; teoremin sol tarafı; eleştirmen güncellemesinde γⁿ; A2C ve DPG sözde kodlarında max.
  - M11.10: planlama çiftleri görülmüş olmalı; "tampon genelleme yapmaz" ifadesinin nüansı; düğüm değeri tanımı; PUCT bonusunda eksik √N.
  - M11.11: RL adımında uzman hamlesi yerine oynanan hamle; değer ağı güncellemesinin işareti; Ke Jie maçı AlphaGo Master'ındır; kazanma kuralında komi.
  - M11.12: Taylor açılımının yazımı; genel algoritmada adım kabulü, oran ölçütü ve sınırda büyütme eksik; belirsiz Hessian'da alt problem global çözülebilir (sezgiseller maliyet içindir), dogleg yalnız $H\succ0$ iken tanımlıdır.
- **Ölçümle düzeltilen ya da anlatıma giren sonuçlar** (ayrıntılar commit mesajlarında):
  - M11.2: yoğun rastgele MDP'lerde açgözlü politika hep ilk yinelemede doğru çıkıyordu; varsayılan halka yapısına alındı.
  - M11.3: PI'ın her ayarda kazanması yoğun matris varsayımından geliyordu; seyrek geçişlerde MPI'nin üstünlüğü göründü.
  - M11.5: çift Q'nun yavaşlığı yanlılıktan değil verinin iki tabloya bölünmesinden geliyor.
  - M11.6: tekrar deneyinde "unutma" yerel özniteliklerle görünmüyordu; küresel Fourier öznitelikleri eklendi.
  - M11.9: DPG gradyanı yalnız başlangıç durumundan hesaplanıyordu; indirimli ziyaret ağırlığı eklendi ve sonlu farklarla doğrulandı.
  - M11.10: "X engellemeli" konum aslında tek hamlede kazandıran çift tehdit konumu; "kötü önsel keşfi yavaşlatır" iddiası ölçümde tutmadı.
  - M11.11: kök arama deneyinde ziyaret edilmemiş çocuklara gerçek Q veriliyordu; AlphaGo Zero'daki gibi 0 yapıldı.
  - M11.12: GAE deneyinin ilk sürümü varyansı tek örnekten ölçüyordu (hata yanlılığa yazılıyordu). Ayrıca kırpma, KL'yi kesin sınırlamıyor: ε = 0.5'te ilk KL 0.124 ölçüldü.
- **Bakım:**
  - M1.3 derinleştirmesi `main`'e alındı. Eski kalan M1.3 notebook'u yenilendi.
  - Rubrik satırlarındaki uzun formüller mobilde taşıyordu; satır metni ayrı bir kutuya alındı.
  - svdlab'de çubuklar üst üste biniyordu; yan yana dizildi.
  - M11.12'nin d aşaması denetimi tek bir tohuma bağlıydı: aynı kod bir makinede PPO için 170, ötekinde 450'den fazla veriyordu. İlk tohum eşiğin altında kalırsa ikinci tohum deneniyor.
  - M11.12 girişindeki "belirsiz Hessian'da yalnız sezgiseller" cümlesi düzeltildi.
- **Kalan:** kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 10

Faz 10'un (paralel programlama ve GPU) 7 modülünün hepsine tam konu anlatımı yazıldı. Kaynak, BİL 455/533 Parallel Processing dersinin dokuz slayt destesi (lect_01–lect_09) ve dört alıştırma föyüdür (Ex. 1–4). Her deste sayfa sayfa işlendi. Slayt metni kopyalanmadı. Konular yeniden anlatıldı, türetimler eklendi ve slaytlardaki hatalar "Kaynak düzeltmesi" kutularında gerekçeleriyle düzeltildi. C++ thread ve CUDA kodları tarayıcıda çalışmadığı için deneyler, ölçüm ya da simülasyonla kurulmuş maliyet modelleridir. Her deneyin sınırları kendi kutusunda yazılıdır.

| modül | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|
| M10.1 paralel modeller | 2 | amdahl, workspan, mandelbal | Amdahl ve Gustafson, iş–açıklık, Graham/Brent sınırları, yük dengeleme ve parça tavanı |
| M10.2 thread'ler ve eşzamanlama | 2 | interleave, deadlock, prodcons | kayıp güncelleme sayımı, kilitlenmenin dört koşulu, bellek modeli ve acquire–release, monitörler |
| M10.3 CPU'da paralel çekirdekler | 2 | histlab, scanlab, matmulcache | blok taraması, integral görüntü geçişleri, evrişim bantları ve hale, matmul döngü sırası ve karolama |
| M10.4 CUDA ve bellek | 2 | occupancy, bankconf, divergence | birleştirme ve sektörler, CGMA tavanı, bank eşlemesi ve dolgu, doluluk, Little yasası |
| M10.5 paralel indirgeme | 2 | redtree, fpsum, coarsen | iş ve adım, denetim ve bellek ayrışması, kabalaştırma koşulu, sıralı/ikili/Kahan toplam hatası |
| M10.6 paralel tarama | 2 | scanalgo, linrec | Kogge–Stone işi ve WAR tehlikesi, Brent–Kung, hiyerarşik ve domino tarama, afin bileşim taraması |
| M10.7 GPU'da evrişim ve matmul | 4 | convtile, convmethods, regtile | halo'lu karonun azalma oranı, Winograd F(2, 3), yazmaç karolaması, CPU–GPU hesap/aktarım oranı |

- **Faz 10'da 20 deney (19'u yeni; M10.1'deki amdahl korundu) ve 16 şekil.** Şekillerdeki bütün sayılar betikle hesaplandı.
- **Kaynak düzeltmeleri** (her biri ilgili modülde kutu içinde):
  - M10.1: Cell'in duyuru yılı ve PS3'te kapalı SPE; 102.4 GFLOPS PowerXCell 8i'nindir; Kepler'de 255 yazmaç thread başına sınırdır; Amdahl yasası güçlü ölçekleme içindir; önek toplamı kodunda `+=` yerine `=`; map/scan/reduce tanımları.
  - M10.2: fork'un dönüş değeri ve yazınca kopyalama; join nesneyi yok etmez; anlatılan şey NUMA değil önbellek hiyerarşisidir; donanım SC değildir; volatile thread'ler arası bayrak değildir; std::atomic'in kapsamı; acquire–release'in anlamı; make_weak ve standart fiber yoktur; thread güvenli ile yeniden girilebilir farklıdır.
  - M10.4: Fermi'nin hesap yeteneği 2.x, 4.x yoktur; thread blok kümeleri Hopper/CUDA 12'dir; A100'de en çok 7 MIG; __syncthreads thread'leri eşzamanlar; ana makine yerel belleğe erişemez; cubin uyumluluğu; shuffle 3.0'da geldi; 8 baytlık bank kipi Kepler'e özgüdür.
  - M10.5: 256 öğede ayrışan yinelemeler yediden beşe iner; yakınsak eşleme değişmelilik ister; warp işlemlerinin tarihleri.
  - M10.6: "bölümlü tarama" bayraklı taramanın adıdır, anlatılan yöntem hiyerarşik taramadır.
  - M10.7: Ex. 3'te GPU maddesine kopyalanmış "standart C++ thread'leri" ifadesi ve 8 bitlik integral dizi çıktısı; Ex. 4'te tamsayı girdinin fp32'de 2^24 sınırı.
- **Ölçümle düzeltilen ya da anlatıma giren sonuçlar:**
  - M10.1: kritik yol önceliği simetrik grafiklerde etkisizdi, düzensiz grafiklere bağlandı; desen–yapı tablosu slayt görüntüsüne göre düzeltildi.
  - M10.2: kilitlenme simülasyonu thread'leri sırayla işlediği için livelock göstermiyordu; istekler eşzamanlı çözülecek biçimde yeniden yazıldı.
  - M10.3: atomik histogram modeli satır transferlerinin serileşmesini içermiyordu; matmul önbellek simülasyonunda n = 64'ün n = 68'den 2.4 kat fazla ıska vermesi 2'nin kuvveti çakışmalarıyla açıklandı ve dayanaksız bir öneri kaldırıldı.
  - M10.4: boş else yolu ayrışmayı sıfır gösteriyordu; "döngü boyunu 32'nin katı yap" önerisi yanlıştı.
  - M10.5: kabalaştırma modeli bant genişliğini bütün SM'ler doluymuş gibi hesaplıyordu.
  - M10.7: evrişim yöntemi karşılaştırmasında ayrılabilir yol her zaman en hızlı seçiliyordu, ayrı raporlanıyor; matmul modelinde çift tampon ve k diliminin etkisi yoktu, bariyer kaybı eklendi.
- **Dağıtım sorunu:** M7.4'ten M10.5'e kadar yapılan değişiklikler, M8.1'in bir pratik kartındaki yuvarlanmış cevap (0.000695) CI modül testini geçemediği için yayına çıkmamıştı. Cevap 0.00069541 olarak düzeltildi; o tarihten bu yana her gönderimden sonra CI'nin ve Pages dağıtımının sonucu denetleniyor.
- **Test ortamı notu:** C++ aşamaları için sunucuya GCC 16 ve ninja kuruldu. Triton yerel olarak kurulu olmadığı için Triton aşamaları yalnız CI'de sınanıyor.
- **Kalan:** Faz 11 (YAP 446 pekiştirmeli öğrenme, DQN ve AlphaGo makaleleri) konu anlatımları; kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 9

Faz 9'un (klasik görüntü ve video işleme) 18 modülünün hepsine tam konu anlatımı yazıldı. Kaynak, BİL 467/561 Image Processing ders arşivindeki 30 slayt destesidir; her deste sayfa sayfa işlendi ve görsel slaytlar sayfa görüntüsüne çevrilerek kontrol edildi. Slayt metni kopyalanmadı. Konular yeniden anlatıldı, türetimler eklendi ve slaytlardaki hatalar "Kaynak düzeltmesi" kutularında gerekçeleriyle düzeltildi. Destekleyici kaynaklar Gonzalez–Woods'tur; özgün makaleler de kullanıldı: Canny, Harris, Lowe, Dalal–Triggs, Viola–Jones, Lucas–Kanade, Bolme (MOSSE), Barnich (ViBe), Wang (SSIM) ve Tomasi (ikili süzgeç). OpenCV, scikit-image ve colorsys ile birebir karşılaştırmalar yapıldı.

| modül | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|
| M9.1 dijital görüntü | 3 | interp, fwdinv | elektromanyetik spektrum ve dalga boyu, örnekleme ve nicemleme, enterpolasyon çekirdekleri, ileri/ters eşleme ve delikler |
| M9.2 yoğunluk ve histogram | 2 | histeq, histmatch | histogram eşitlemenin CDF türetimi, eşleştirme ve monotonluk |
| M9.3 uzamsal süzgeçleme | 2 | smoothcmp, padmodes | korelasyon/evrişim, ayrılabilirlik ve maliyeti, kutu ile Gauss, dolgu kipleri |
| M9.4 keskinleştirme | 2 | deriv1d, sharpen1d | birinci/ikinci türev davranışı, Laplace ve keskinleştirme maskesi, yüksek artırma |
| M9.5 restorasyon | 2 | noisepdf, denoise2d | altı gürültü yoğunluğu ve momentleri, ortalama ve sıra istatistiği süzgeçleri, uyarlamalı medyan |
| M9.6 kenarlar | 3 | edgeop, cannylab | gradyan operatörleri, LoG ve sıfır geçişleri, Canny'nin dört adımı |
| M9.7 köşeler | 2 | harrisgeo, harrismap | yapı tensörü, özdeğer geometrisi, Harris yanıtı |
| M9.8 morfoloji | 3 | morphbin, morphgray | genişleme/aşınma/açma/kapama, dualite, gri seviye morfoloji |
| M9.9 renk | 2 | colorconv, colorproc | RGB/HSI/HSV/Lab dönüşümleri, renk histogram eşitleme, Di Zenzo gradyanı |
| M9.10 frekans uzayı | 3 | circconv, freqfilt, deconv | DFT ve dairesel evrişim, ideal/Butterworth/Gauss süzgeçler, Wiener süzgeci |
| M9.11 bölütleme | 3 | pointline, segcluster, slic | nokta/çizgi tespiti, k-means ve GMM, SLIC uzaklığı |
| M9.12 kenar koruyan süzgeçler | 3 | bilateral1d, clahelab, nlmlab | CLAHE kırpma ve dağıtma, ikili süzgeç, yerel olmayan ortalamalar |
| M9.13 kalite ölçütleri | 2 | eqmse, ssimparts | MSE/PSNR'nin kandırılması, SSIM'in l/c/s parçaları ve C3 = C2/2 |
| M9.14 öznitelikler | 3 | dogscale, hoglab | ölçek uzayı, DoG ≈ (k − 1)σ²∇²G, SURF kutu süzgeçleri, SIFT ve HOG betimleyicileri |
| M9.15 eşleştirme | 2 | ncclab, ransaclab, mossetrack | korelasyon katsayısı, homografi ve DLT, RANSAC iterasyon formülü, MOSSE |
| M9.16 Viola–Jones | 2 | haarfeat, adaboostlab | integral görüntü, 162 336 özniteliğin sayımı, VJ AdaBoost'unun özgün AdaBoost'a denkliği, kaskad aritmetiği |
| M9.17 video ve hareket | 3 | lkflow, blockfruc, stablab | optik akış denklemi ve normal akış, LK ve yapı tensörü, piramit, ileri–geri denetimi, yumuşatmanın frekans yanıtı |
| M9.18 arka plan çıkarma | 2 | bgsub, vibeghost | üstel ortalamanın belleği, Stauffer–Grimson, ViBe örnek ömrü, doğrusal özyinelemenin paralel taraması |

- **Faz 9'da 41 deney (40'ı yeni; M9.12'deki bilateral1d korundu) ve 44 şekil.** Şekillerdeki bütün sayılar betikle hesaplandı.
- **Birebir doğrulanan deneyler:**
  - clahelab, `cv2.createCLAHE` ile 2 × 2'de birebir aynı; 8 × 8'de 57 pikselde 1 gri seviye fark var (yuvarlama kuralı).
  - colorconv'un Lab dönüşümü scikit-image, HSV dönüşümü colorsys ile aynı.
  - eqmse'nin SSIM'i scikit-image ile 5 basamak aynı.
  - hoglab, `skimage.feature.hog` ile 10⁻⁷ içinde aynı.
  - ncclab, `cv2.matchTemplate(TM_CCOEFF_NORMED)` ile 10⁻⁵ içinde aynı.
  - M9.16'daki öznitelik sayısı `haar_like_feature_coord` ile aynı (162 336).
- **Kaynak düzeltmeleri** (slayt görüntüleriyle doğrulandı; her biri ilgili modülde kutu içinde):
  - M9.1: görünür bant µm, ultrason dalga boyu, afin dönüşümün koruduğu nicelikler.
  - M9.2: 8 bitte L = 256, monotonluğun anlamı.
  - M9.3: ayrılabilirlik tanımı, 11 × 11 çekirdeğin kazancı.
  - M9.5: Erlang'ın tanım aralığı, tuz-biber momentleri ve yer değiştirmiş etiketler.
  - M9.6: g_y, LoG paydası σ⁴, sıfır çemberi √2σ, Canny açı ve zayıf eşik aralıkları.
  - M9.7: R'nin tanımı, elips eksenleri √λ.
  - M9.8: genişleme ve kapama küme formülleri.
  - M9.9: hayali CIE ana renkleri, b* ekseni, yayın yılı.
  - M9.11: k-means amacı kareler toplamıdır.
  - M9.13: S ≤ 1.
  - M9.14: Gauss'un tekliği kuramsaldır; iki bölmeye enterpolasyon; Frobenius normu; HOG maskesi ve L2-Hys.
  - M9.16: dört dikdörtgenli öznitelik çaprazdır; VJ başlangıç ağırlıkları; aydınlatma dayanıklılığının kaynağı.
  - M9.17: LK türetimindeki yazım ve işaret hataları, ileri–geri denetiminin karşılık noktası, kayan deklanşörde satırlar, homografinin 8 serbestliği.
  - M9.18: ViBe'de komşu güncellemesi bağımsız bir çekiliştir.
- **Ölçümle düzeltilen ya da anlatıma giren sonuçlar:**
  - Taslak notlarda ölçümle çürütülüp yeniden yazılan iddialar:
    - harmonik ortalamanın Gauss gürültüsünde iyi olduğu;
    - Gauss çekirdeğinin dairesel sarmada daha az etkilendiği;
    - koşulsuz "NLM dokuyu korur" (bu sahnede ikili süzgecin yöntem gürültüsü 0.009, NLM'ninki 0.025);
    - HOG normalizasyonunun kosinüs benzerliğini artırdığı (yerine şablon skor oranı ölçüldü);
    - eşit MSE görüntülerinde SSIM sıralamasının Wang ve ark. ile aynı olduğu;
    - RANSAC'ta nominal ε ile hesaplanan başarı olasılığı (gerçekleşen içerdeğer oranı kullanıldı);
    - MOSSE'nin tek örnekle başlatılması.
  - M9.12: CLAHE'de OpenCV ile bir gri seviyelik fark, yarıyı çifte yuvarlama ile giderildi.
  - M9.14: mini SIFT en küçük lekeyi kaçırıyordu, 2 kat büyütme seçeneği eklendi; en büyük leke z = 1.5'te kaçıyordu, 4 oktav kullanıldı.
  - M9.16: kaskadın ilk sürümü tek AdaBoost'un önekleriydi. 0/1 kütük çıktıları yüzünden ilk aşamalar hiçbir şey reddetmiyordu; zor negatif madenciliğiyle ayrı eğitilen aşamalara geçildi. Zor negatifler eklenmeden görev çok kolaydı (test FPR 0).
  - M9.17: FRUC deneyinin varsayılan hızı arama aralığını aşıyordu.
  - M9.18: deney videosundaki duran nesne yanlış işaretli bir min/max yüzünden durmuyordu.
- **Pratik kartlarında düzeltilen hesap hataları** (yayından önce, test yakaladı): M9.4 p4, M9.7 p3, M9.9 p2/p3/p6, M9.10 p2, M9.11 p4, M9.12 p4, M9.13 p4, M9.15 p4, M9.17 p4 (yuvarlanmış cevap).
- **Şekil sorunları** (ekran görüntüsünde fark edildi):
  - SVG metninde ham "<" karakteri;
  - tanımsız bir dolgu sınıfının siyah çizilmesi;
  - renk adlarıyla paletin uyuşmadığı başlıklar (M9.16 histogramı, M9.17 LK);
  - çakışan alt yazılar;
  - görüntü üstü işaretlerde ters y ekseni.
- **Kalan:** Faz 10 (BİL 455) ve Faz 11 (YAP 446 ve DQN/AlphaGo) konu anlatımları; kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 8

Faz 8'in iki proje modülüne tam konu anlatımı yazıldı. İkisi de önceki fazların araçlarını tek bir çalışan sistemde birleştiriyor; anlatım projenin her adımını (bütçe, biçim, eğitim, değerlendirme, profil) gerekçeleriyle veriyor.

| modül | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|
| M8.1 mini-LLM | 2 | budget, mfu | 6ND (ileri 2N, geri 4N), Chinchilla eş-FLOP eğrisi, ters cevabın gerekçesi, sabit tahmin tabanı (1.766 nat/konum), ısınma + kosinüs, MFU |
| M8.2 bitirme projesi | 2 | emalab, pairboot | difüzyon U-Net'inin parçaları, EMA penceresi 1/(1 − β), CFG'nin NFE maliyeti, Dice, eşleştirilmiş bootstrap'ın varyans kazancı √(1 − ρ) |

- **Ölçümle düzeltilen ya da anlatıma giren sonuçlar:**
  - M8.1: Chinchilla yaklaşım 3 katsayılarının "her ölçekte daha büyük model önerdiği" ilk notu yanlıştı; 10⁷ parametrede biraz büyük, Chinchilla bütçesinde (6·10²³ FLOP) 3.2·10¹⁰ ile daha küçük çıktı (makale yaklaşım 3 için 40 milyar bildirir). Not ölçeğe göre koşullu yazıldı. MFU deneyinde GPT-2 small yapılandırması N ≈ 1.243·10⁸ veriyor (gerçek 124 439 808).
  - M8.2: EMA deneyinde "yarı yola gecikme" ilk tanımı (0.5 eşiği) büyük β'da anlamsızdı; sıçrama anındaki değerden ölçülecek biçimde değiştirildi ve gecikmenin parametrenin kendi yavaşlığından ve EMA'dan gelen iki parçası ayrı yazıldı.
  - M8.1 Şekil 1'de tırnaksız bir SVG özniteliği satırları kesiyordu (ekran görüntüsünde fark edildi).
- **Bilinen belirsizlik:** sunucudaki başsız tarayıcıda Literata yazı tipi yüklenemediği için birleştirici aksanlı harfler (x̂, θ̄) eski deneylerin bazı etiketlerinde kutu olarak görünüyor; şekillerde aksan ayrı glif olarak bindirildi, deney metinlerinde gerçek tarayıcılarda sorun olmaması beklenir ama doğrulanmadı.
- **Kalan:** Faz 9–11 konu anlatımları (BİL 467, BİL 455, YAP 446 kaynaklarıyla); pratik kartları; kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 7

Faz 7'nin (donanım, ölçekleme ve dağıtım) 6 modülünün hepsine tam konu anlatımı yazıldı. Ders arşivinde bu konulara slayt yok; kaynaklar PMPP, GPU MODE dersleri, Triton eğitimleri, orijinal makaleler (Micikevicius ve ark. 2018/2022, Kalamkar ve ark. 2019, Rajbhandari ve ark. 2020, Li ve ark. 2020, Zhao ve ark. 2023, Jacob ve ark. 2018, Dettmers ve ark. 2022, Frantar ve ark. 2022, Han ve ark. 2015) ve PyTorch belgeleridir. Davranış iddiaları PyTorch 2.15'te çalıştırılarak doğrulandı (ör. torch.export'un veriye bağlı if'te GuardOnDataDependentSymNode vermesi, jit.trace'in sessizce yanlış graf üretmesi).

| modül | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|
| M7.1 CUDA | 2 | coalesce, tilecount | global indeks ve sınır kontrolü, adımlı erişimin verimi, döşemenin 2n³/T kuralı ve tile yoğunluğu T/e |
| M7.2 Triton | 2 | tritongrid, l2swizzle | maske ve other, birleşik softmax 2N/6N, gruplu sıranın L2 paylaşımı (90 → 54) |
| M7.3 karışık hassasiyet | 2 | floatfmt, lossscale, stochround | bitlerden aralık/eps, kaybolan güncelleme, kayıp ölçekleme, stokastik yuvarlamanın yansızlığı ve bit hilesi |
| M7.4 dağıtık eğitim | 2 | ringar, zeromem | gradyan ortalaması = büyük batch, ring 2(N − 1)/N ve alt sınırı, ZeRO aşamaları ve 2Ψ/3Ψ |
| M7.5 nicemleme ve budama | 2 | quantlab, prunelab | s²/12 hatası, ölçek kapsamı ve 16/g, int4 paketleme, budamada atılan enerji |
| M7.6 dışa aktarma | 2 | tracetrap, batchserve | izlemenin kontrol akışını dondurması, torch.cond, sabit katlama, dinamik batch'lemenin kapasite sınırı |

- **Faz 7'de 13 yeni deney.** l2swizzle Triton eğitimindeki 90/54 sonucunu birebir üretiyor; floatfmt bütün el hesabı alanlarını doğruluyor.
- **Ölçümle düzeltilen ya da anlatıma giren sonuçlar:**
  - M7.1: tilecount'ta shared memory sınırı kontrolü ilk sürümde çıktıyı etkilemiyordu (lab testi yakaladı); sınırın izin verdiği en büyük T eklendi. fp32'de thread başına bir çıktıyla tile yoğunluğunun A100 dirseğine ulaşamadığı (T ≤ 32 → 8 < 10) yazmaç döşemesinin gerekçesi olarak verildi.
  - M7.3: stokastik yuvarlamanın ilk sürümü 2'nin kuvveti sınırlarında yanlış komşuyu seçebiliyordu; yeniden yazılıp 20 000 örnekle yansızlığı sınandı.
  - M7.4: α–β modelinde küçük mesajlarda (1 MB, α = 100 µs, 16 işçi) merkezi yaklaşımın ring'den hızlı çıktığı görüldü; not koşullu yazıldı.
  - M7.6: KaTeX `B_\max` yazımını reddetti (site testi yakaladı); `B_{\max}` olarak düzeltildi.
- **Pratik kartlarında düzeltilen hesap hataları** (yayından önce): M7.4 p3 (ZeRO-2: 9.5 GB) ve M7.2 p3'teki yanlış-cevap mesajı.
- **Kalan:** Faz 8–11 konu anlatımları; pratik kartları; kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 6

Faz 6'nın (PyTorch'un içi) 5 modülünün hepsine tam konu anlatımı yazıldı. Ders arşivinde bu konulara slayt olmadığı için kaynaklar PyTorch belgeleri (BSD-3), JAX Autodiff Cookbook, orijinal makaleler (Chen ve ark. 2016, Korthikanti ve ark. 2022, Abadi ve ark. 2016, Ansel ve ark. 2024, Bengio ve ark. 2013, Ganin–Lempitsky 2015) ve **PyTorch 2.15'te yapılan ölçümlerdir**. Anlatımdaki her davranış iddiası (neyin saklandığı, hangi in-place işlemin hata verdiği, checkpoint'te BN'nin iki kez güncellenmesi, graph break sayısı) çalıştırılarak doğrulandı.

| modül | şekil | deneyler | öne çıkan türetimler ve ölçümler |
|---|---|---|---|
| M6.1 autograd'ın içi | 2 | autogradsim, savedmem | VJP ve dallanmada toplama, bağımlılık sayan motor, saklanan tensör tablosu, aynı depo bir kez (ReLU 6d, GELU 10d, dropout 14d; ölçüldü) |
| M6.2 özel Function | 2 | fdcheck, stelab | lse ve p − y gradyanları, sonlu farkın h²/6 + u/h hata analizi, STE ve GRL; birleşik CE yarı bellek (ölçüldü) |
| M6.3 torch.func | 2 | jaccost, dpsgd | dual sayılarla JVP, (1 + n)F ile (1 + m)F, örnek başı gradyan = dış çarpım, DP-SGD, HVP = gradyanın JVP'si |
| M6.4 bellek | 2 | ckptsim, accumlab | sbh(34 + 5as/h), √n checkpointing, n_j/N ve T_j/T biriktirme ölçekleri |
| M6.5 torch.compile | 2 | roofline, fusegroups | roofline ve dirsek, füzyonun bayt kazancı, Conv + BN katlama, Dynamo / AOTAutograd / Inductor |

- **Faz 6'da 10 yeni deney.** autogradsim'in sonuçları ve hata metinleri PyTorch'la birebir; jaccost Jacobian'ı iki yoldan gerçekten hesaplıyor (fark 10⁻¹⁶).
- **Ölçümle düzeltilen ya da anlatıma giren sonuçlar:**
  - M6.1: GELU + dropout bloğunun belleği ilk tahminde 18d'ydi; dropout'un çarpma düğümü yalnız maskeyi sakladığı için ölçüm 14d verdi.
  - M6.2: "GRL'de girdiyi aynen döndürmek düğüm bağlamaz" iddiası PyTorch 2.x'te geçerli değil (iki biçim de aynı gradyanı verdi). Aşama e'nin "naif SiLU üç tensör saklar" ifadesine not eklendi: 3 referans ama 2 ayrı depo, kazanç yarı.
  - M6.3: DP-SGD deneyinde gürültünün doğruluk dalgalanmasını artıracağı tahmin edilmişti; ölçümde SGD daha dalgalı çıktı (0.023'e karşı 0.0008), çünkü aykırı örneklerin büyük gradyanları kırpılıyor. %30 aykırı örnekte SGD 0.40, DP-SGD 0.93.
  - M6.4: checkpoint içindeki BN'nin num_batches_tracked sayacı geri geçişten sonra 2 (koşu istatistikleri iki kez güncelleniyor).
  - M6.5: softmax'ın füzyonsuz trafiği ilk sürümde 8N yazılmıştı; doğrusu 6N. torch.compile ile aşama e'deki fonksiyonda 1 graph break, 2 graf ölçüldü.
- **Şekil yazım betiği:** alt simge dönüşümü kod tanımlayıcılarını (add_(1), no_grad, alt_katman) bozuyordu; kural tek harfli tabanlarla sınırlandı ve M4.14'teki "alt_katman(x)" düzeltildi.
- **Kalan:** Faz 7–11 konu anlatımları; Faz 7–11 modüllerine pratik kartları; kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 5

Faz 5'in (üretken modeller, graf ağları, pekiştirmeli öğrenmeye giriş) 8 modülünün hepsine tam konu anlatımı yazıldı. Kaynaklar Derin Öğrenme dersinin ch_17 slaytları (otomatik kodlayıcılar, VAE, GAN, difüzyon, yönlendirme; slayt 2–132'nin tamamı), YAP 446 lect_1 ve lect_8 (M5.5) ve orijinal makalelerdir. M5.4 için ders arşivinde GNN slaytı olmadığından açık literatür (Kipf–Welling 2017, Xu ve ark. 2019, Hamilton ve ark. 2017, Veličković ve ark. 2018) kullanıldı.

| modül | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|
| M5.1 VAE | 3 | klasym, gradest, betavae | KL ≥ 0, ileri/ters KL, ELBO'nun iki türetimi, Gauss KL kapalı formu, reparametrizasyon, norm yoğunlaşması, doğrusal VAE çöküş eşiği |
| M5.2 GAN | 3 | diracgan, gan1d | V(D, G) = negatif BCE, D* ve JS, doymayan kaybın gradyanı, Dirac GAN özdeğerleri, WGAN |
| M5.3 difüzyon | 3 | diffsched, diff1d | varyansın 1'e yakınsaması, q(x_t | x₀) kısa yolu, NLL sınırının KL'lere ayrılması, q(x_{t−1} | x_t, x₀), L_simple, Tweedie |
| M5.4 GNN | 2 | oversmooth, wltest | permütasyon eş-değişkenliği, GCN'in spektral türetimi ve yeniden normalizasyon, aşırı düzleşme, 1-WL üst sınırı, GIN |
| M5.5 RL | 2 | bellman2, reinforce | Bellman denklemi ve (I − γP)⁻¹r, değer iterasyonunun γ-büzülmesi, politika gradyanı teoremi, taban çizgisinin yansızlığı |
| M5.6 otomatik kodlayıcılar | 2 | dae1d, sparsecode | doğrusal AE = PCA, bağlı ağırlık sayımı, Tweedie (DAE artığı = σ²·skor), L1 yumuşak eşikleme, KL seyreklik |
| M5.7 GAN II | 3 | fidis, trunclab | IS = exp I(x; y), 1-B Fréchet uzaklığı, eşitlenmiş öğrenme oranı, AdaIN, StyleGAN2 demodülasyonu |
| M5.8 yönlendirme | 3 | cfg2d, guide1d | Bayes'ten sınıflandırıcı yönlendirmesi, CFG = kesin sınıflandırıcı yönlendirmesi, Gauss eğimi, SNR ve kosinüs takvimi |

- **Faz 5'te 15 yeni deney** (cfg2d mevcuttu). Ön ayar notlarındaki bütün sayılar Node'da ölçüldü.
- **Ölçümle ortaya çıkan ve anlatıma giren sonuçlar:**
  - M5.7: 2 boyutlu öznitelik uzayında FID, sınıf başına tek örnek üreten bir üreticiyi neredeyse görmüyor (0.017; gerçek–gerçek tabanı 0.025). Sınıf içi değişim ayrı boyutlarda yaşayınca görüyor (6.0). Not iki durumu ayrı açıklıyor. Aynı deneyde IS bu üreticide gerçek veriden yüksek (9.78 > 8.44).
  - M5.7: kusurlu üreticide kesinlik 0.96'dan 0.72'ye düşerken 2-B FID 0.014'ten 0.015'e çıkıyor; kesinlik/geri çağırmanın gerekçesi olarak verildi.
  - M5.8: kesin skorlarla bile s = 3 CFG örneklerinin ortalaması 1.61, std'si 0.355; yönlendirmenin "hedeflediği" p(x)p(y|x)³ ise 1.11 / 0.522 (200 adımda aynı). CFG'nin eğik dağılımdan tam örneklemediği bir ayrıntı kutusuyla açıklandı.
  - M5.6: seyrek kod deneyinde ISTA 800 adımda yakınsamıyordu (FISTA'ya geçildi); zıt işaretli yakın atomlar desteği bozduğu için "en küçük atom aralığı" kontrolü eklendi ve aralık 0'da kurtarmanın çöktüğü bir senaryo olarak gösterildi.
  - M5.3: T = 20'de β = 1 sıfıra bölmeye yol açıyordu; β 0.5 ile sınırlandı. M5.2: doymayan kaybın işareti düzeltildi.
- **Ders slaytlarındaki hatalar ve notlar (anlatımda "Kaynak düzeltmesi/notu" kutusuyla):**
  - ch_17: kodlayıcı/kod çözücü parametre gösterimi literatürün tersi; Z ~ N(μx + μy, σy² + σy²) → σx² + σy²; "sinyal doğrusal azalır" → doğrusal olan β; iki ayrı Nichol–Dhariwal makalesi; "Detector network" → discriminator; "tek Nash dengesi" sonucunun koşulları (birkaç modülde); "Tero Kerras" → Karras; IS'in çeşitlilik ölçütü ortalama sınıf dağılımının entropisidir; ProGAN SWD kullanır, StyleGAN'a geçiş kitabın anlatı köprüsüdür; slayt 28'deki bağlı ağırlık gözlemi genel kural değildir; seyreklik için yalnız L1 vardı, KL eklendi; CFG "çapraz gömme uzayları" gerektirmez; slayt 131'deki "toplamı girdiden çıkar" kısaltması ve sınıflandırıcının gürültülü veride eğitilmesi gerekliliği.
  - YAP 446: "Reinforced learning" → reinforcement; Atari ödülü skor değişimidir; 3³⁶¹ ≈ 1.74 × 10¹⁷²; "negatif pekiştirme" psikolojideki terimle uyuşmuyor.
- **Şekillerde yazım düzeltmesi (tüm modüller):** 16 modülün şekillerinde SVG metnine düz yazılmış x_t, x_{t−1}, F_(m−1) gibi alt simgeler ve D^(−½) gibi üst simgeler gerçek alt/üst simgeye çevrildi. Yazı tipinde birleşik glifi olmayan aksanlı harfler (θ̂, x̂, X̃, D̃, h̄, w̄) kutu olarak çiziliyordu; aksan işareti harfin üstüne ayrı bir glif olarak bindirildi. Şekil altyazılarındaki alt simgeler `<sub>`'a çevrildi.
- **Erişim sınırı:** slaytlardaki görüntüler (MNIST yeniden yapılandırmaları, GAN örnekleri, StyleGAN yüzleri, difüzyon örnekleri) metne dökülemedi.
- **Kalan:** Faz 6–11 konu anlatımları; Faz 6–11 modüllerine pratik kartları; kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 4

Faz 4'ün (derin öğrenme mimarileri) 16 modülünün hepsine tam konu anlatımı yazıldı. Kaynak, Derin Öğrenme dersinin ch_14, ch_15, ch_16, ch_17 (Faz 5'e kalan kısmı hariç), embeddings_updated ve contextualizedEmbeddings slaytlarıdır. Slaytların her maddesi yeniden yazıldı. Orijinal makalelerdeki (Sutskever 2014, Bahdanau 2015, Luong 2015, Vaswani 2017, Mikolov 2013a/b, Levy–Goldberg 2014, GloVe, fastText, ViT, ELMo, GPT-1, BERT, InstructGPT, DPO …) sayılar makale metinlerinden doğrulandı ve daha ayrıntılı türetimler eklendi.

| modül | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|
| M4.1 konvolüsyon | 2 | conv2dviz, convcost, im2colviz, poolshift | kaymaya eşdeğişirlik, çıktı boyu, col2im eşleniği, max havuz geri geçişi |
| M4.2 CNN mimarileri | 3 | rfcalc, resgrad | küçük çekirdek eşdeğerliği, alıcı alan yinelemesi, artık yolun gradyanı |
| M4.3 RNN | 2 | rnnunroll, bpttsum | BPTT, kesilmiş BPTT, doğrusal rekürsiyonun paralel taraması |
| M4.4 LSTM / GRU | 2 | gatemem, lstmgrad | hücre yolu gradyanı ve sabit hata karuseli, PyTorch ağırlık düzeni |
| M4.5 embedding ve BPE | 3 | embedgeom, bpeviz | one-hot geometrisi, embedding = one-hot × W, bigram = sayım MLE'si |
| M4.6 dikkat | 3 | attnviz, onlinesoftmax (+ attnscale) | √d_k ölçeklemesi, çevrimiçi softmax |
| M4.7 GPT | 4 | gptcount, sampler, lnplace, kvcache | parametre/FLOP sayımı, Pre-LN akışı, KV önbelleği |
| M4.8 CNN blokları | 4 | blockcost, sepkernel, camviz | 1×1 darboğaz, derinlemesine ayrılabilir oran 1/C_out + 1/K², SVD ile ayrılabilirlik, CAM |
| M4.9 tespit | 4 | nmsviz, apcurve, anchorkm | √w kaybı, yoğun katman = valid konvolüsyon, AP zarfı, RoI seviye ataması |
| M4.10 bölütleme | 4 | tconv1d, upsample2d, segmetrics | 2× bilineer çekirdek, transposed konvolüsyonun dört görünümü, U-Net 572 → 388, Dice = 2J/(1 + J) |
| M4.11 zaman serisi | 3 | armasim, tsforecast, wavenetrf | AR durağanlığı ve varyans, Yule–Walker, fark almanın eğilimi silmesi, WaveNet alıcı alanı |
| M4.12 word2vec | 3 | sgnsgrad, noisedist, pmisvd, w2vlab | OOV çakışma beklentisi, tam softmax gradyanı, dinamik pencere ağırlığı, SGNS gradyanları, alt-örneklemenin √(tf) sıkıştırması, SGNS = PMI − log k, GloVe amacı, 3CosAdd |
| M4.13 seq2seq | 4 | padwaste, sampledsm, beamlab, alignattn | öğretmen zorlaması = MLE, maskeli kayıp ortalamaları, örneklenmiş softmax'ın log(m·q) düzeltmesi, hata birikimi ε·T(T+1)/2, Bahdanau/Luong denklemleri |
| M4.14 Transformer | 3 | maskviz, posenc, noamlr, labelsmooth | Tablo 1 karmaşıklıkları, −∞ maskeleme, base parametre sayımı, permütasyon eş-değişkenliği, P_(p+k) = R_k P_p, √d ölçeği, Noam tepesi, etiket yumuşatma optimumu |
| M4.15 ViT | 2 | patchviz, posinterp, swinreach | yama embedding'i = adımı P konvolüsyon, ViT-B/16 sayımı, interpolasyon koordinatı, Swin maliyeti |
| M4.16 ön eğitimli LM | 3 | noiselab, rlhfdpo (+ softmaxT) | biLM amacı, çift yönlü dikkatte kopyalama sorunu, BERT sayımı, 12Ld², Bradley–Terry, KL cezalı optimum, DPO kaybı |

- **Faz 4'te toplam 51 deney**; her birinin hazır senaryo notlarındaki sayılar Node'da ölçülerek yazıldı. Elle yazılan gradyanlar (alignattn, sgnsgrad, pmisvd) sayısal türevle doğrulandı.
- **Deney sonucuna göre düzeltilen ya da eklenen anlatımlar:**
  - M4.13: örneklenmiş softmax'ta yalnız log q düzeltmesi (ek x3'teki biçim) sık bir hedefte yanlılığı m büyüdükçe sıfıra indirmiyordu (en sık hedefte 0.13'te takıldı). Doğru düzeltme log(m·q) olarak türetildi, deneye seçenek olarak eklendi (yanlılık 0.015).
  - M4.13: beam deneyinin ilk sürümünde T'de kesilen hipotezler tamamlanmış sayılıyordu ve beam "kesin optimumu geçiyordu". Son adımda yalnız EOS'a izin verilerek beam ile Viterbi aynı kurala bağlandı.
  - M4.14: "yalnız sinüs" seçeneği tüm vektörde konumları hâlâ ayırt ediyordu (en yakın iki konumun uzaklığı 2.84, sin + cos'ta 3.71). Not, asıl kaybın tek frekanstaki yansıma konumu ve iç çarpımın p'ye bağlı hâle gelmesi olduğunu söyleyecek biçimde yeniden yazıldı.
  - M4.14: etiket yumuşatma deneyinde iki rejim ölçüldü. Düz CE iyi kalibre iken yumuşatma NLL'yi kötüleştiriyor (0.125 → 0.331); aşırı uyumda iyileştiriyor (13.0 → 4.1). Not ölçülen ortalama güvene göre dinamik.
  - M4.12: alt-örneklemede makaledeki formül ile word2vec.c'deki formülün farkı ve atmanın başladığı φ²t eşiği türetildi.
- **Ders slaytlarındaki hatalar (anlatımda "Kaynak düzeltmesi" kutusuyla):**
  - M4.2: AlexNet'in "%17'ye karşı %26" sonucu iki ayrı sonucun karışımı (ILSVRC-2012: %15.3 / %26.2); LRN hiperparametreleri makaleyle uyuşmuyor.
  - M4.5: one-hot uzaklığı 2 değil √2 (2, karesi).
  - M4.7: zaman çizelgesi ("ChatGPT-1/2/3" → GPT-1/2/3; GPT-3.5'in parametre sayısı açıklanmadı; Mistral ve Qwen 2023; Gemma 3 Mart 2025; "Antropic", "Opes" yazımları); LayerNorm'un öğrenilen parametreleri ortalama/std değil d boyutlu γ, β.
  - M4.8: "ayrılabilir konvolüsyonu varsayılan kullan" önerisine koşullar (FLOP ≠ süre); Inception (3a) "159K" = 163 328/1024.
  - M4.9: `MeanIoU` bölütleme ölçütüdür; YOLOv3'e atfedilen "5 kutu + hücre başına 20 sınıf" düzeni v1/v2 karışımı; lojistik yalnız merkez ofsetlerinde; k-means uzaklığı 1 − IoU; çok ölçekli eğitim 320–608; "25 fps" genel değil.
  - M4.10: "batch > 4 ise BN" kuralına GN makalesinin sayılarıyla not.
  - M4.11: ARMA tarihçesi (Wold 1938, Whittle 1951, Box–Jenkins 1970); deterministik mevsimsellik durağanlığı bozar.
  - M4.12: "Milokov" / "Tomaš" → Tomáš Mikolov; "capitol" → capital; Öklid uzaklığı iddiasına kosinüs notu.
  - M4.13: dolgu örneği [7, 12, 5, 40] için "84 gerçek / 76 dolgu (%50)" → 64 / 96 (%60); "tek nöronlu yoğun katman" hizalama modeli kod çözücü durumunu softmax'ta sadeleştirir (türetim + deney: α satırları özdeş); "negative sampling (or sampled softmax)" iki ayrı yöntem; Luong'un concat sonucu basitleştirilmiş bir uygulamaya ait.
  - M4.14: dikkat tablosunda 0.88 → 0.87 ve 0.94 → 0.95; "38 kelime arayla her zaman aynı" → dalga boyu 37.97, yaklaşık; "yalnız sinüsle 25 ve 35 ayırt edilemez" slaytın parametreleriyle tutmuyor (25 / 32); "slayt 120" atfı → 58; "keyfi uzun cümlelere genişler" → makale "izin verebilir" diyor.
  - M4.16: ELMo'nun girdisi karakter CNN'i (word2vec değil); GPT-1 ≈ 117M (110M BERT-base'in); ChatGPT 30 Kasım 2022; "GPT-5 olacak" sayfası eskimiş (Ağustos 2025'te yayımlandı).
- **Diğer düzeltmeler:**
  - M4.9 `intro.md` ve el hesabı etiketi (YOLO sürümleri).
  - M4.16 `intro.md`'deki "ChatGPT (Aralık 2022)".
  - Yanlışlıkla commit'lenmiş `undefined/` dizini (M3.6 ekran görüntüleri) kaldırıldı.
- **CI eşikleri:** M4.8 `d.check.py` (`cnn2 > fc3 + 0.05`, `cnn2 > fc1 − 0.01`) ve M4.10 `d.check.py` (U-Net mIoU > bilineer + 0.05) iş parçacığı sayısına duyarlıydı: aynı tohumla 1 ve 4 iş parçacığında U-Net 0.807 / 0.722. Eşikler gevşetildi; hata mesajları ölçülen değeri içeriyor.
- **Erişim sınırı:** slaytlardaki şekiller (mimari çizimler, PCA ve konum kodlaması görselleri, dikkat haritaları, GPT-3 grafikleri) metne dökülemedi; içerikleri slayt metninden ve orijinal makalelerin şekil açıklamalarından yeniden kuruldu.
- **Kalan:** Faz 5–11 konu anlatımları; Faz 5–11 modüllerine pratik kartları; kapsam tablosunun final denetimi.

## 2026-11 (3. tur, devam) · Konu anlatımları: Faz 2 ve Faz 3

Aynı düzen (problem → sezgi → tanım → koşullu türetim → sayısal örnek → şekil → deney → PyTorch karşılığı → özet) Faz 2 (klasik ML) ve Faz 3'ün (sinir ağı temelleri) 17 modülüne uygulandı. Faz 3'te ders slaytlarının (Derin Öğrenme, bölüm 10–11) her maddesi anlatıma işlendi; slaytta olmayan ama açık kaynaklarda (orijinal makaleler, Goodfellow ve ark., CS231n) daha ayrıntılı bulunan türetimler eklendi.

| modül | bölüm | şekil | deneyler | öne çıkan türetimler |
|---|---|---|---|---|
| M2.1 k-NN | 8 | 1 | 2-B karar bölgeleri, yüksek boyutta uzaklıkların yoğunlaşması | Voronoi, Cover–Hart sınırı R* ≤ R₁ₙₙ ≤ 2R*(1 − R*), uzaklıkların göreli yayılımının 1/√d ile daralması |
| M2.2 Naive Bayes | 10 | 0 | Gauss NB ve kalibrasyon, yüksek boyutta NB vs LR öğrenme eğrisi | ortak varyansta doğrusal sınır, Laplace düzeltmesi |
| M2.3 SVM | 8 | 1 | dual koordinat inişiyle 2-B SVM | marj = 2/‖w‖, KKT ve destek vektörleri, hinge kaybı |
| M2.4 karar ağacı | 7 | 2 | eksene paralel bölmeler ve derinlik | Gini/entropi kazancı, bölme seçimi |
| M2.5 topluluklar | 7 | 1 | adım adım gradient boosting, bagging varyansı | ρσ² + (1 − ρ)σ²/B, pseudo-residual = negatif gradyan |
| M2.6 PCA / SVD | 8 | 0 | 2-B PCA, kuvvet iterasyonu, SVD ile görüntü sıkıştırma | en büyük varyans yönü (Lagrange), Eckart–Young |
| M2.7 k-means | 8 | 1 | k seçimi (dirsek, siluet) + mevcut k-means | Lloyd adımlarının türetimi ve sonluluğu |
| M2.8 GMM / EM | 9 | 1 | 2-B EM, olabilirliğin monotonluğu | ELBO alt sınırı, E/M adımları, k-means'in sert limiti |
| M3.1 micrograd | 7 | 0 | ifade grafiği (kendi ifadeni yaz, adım adım geri), 2 ay verisinde MLP | yerel türevler, topolojik sıra, gradyan biriktirme |
| M3.2 MLP ve geri yayılım | 7 | 1 | doyan aktivasyonla yavaş öğrenme | δ özyinelemesi, CE + sigmoid'de σ′'nin sadeleşmesi |
| M3.3 tensör autograd | 7 | 0 | yayın (broadcast) görselleştirici, ileri vs ters mod maliyeti | VJP olarak geri kurallar, unbroadcast |
| M3.4 optimizer'lar | 10 | 0 | öğrenme oranı takvimleri (+ mevcut 2-B optimizer, EMA) | momentum, Nesterov, AdaGrad/RMSProp/Adam yanlılık düzeltmesi, AdamW |
| M3.5 başlatma ve aktivasyon | 5 | 0 | aktivasyon istatistikleri (+ mevcut başlatma × aktivasyon) | Glorot/He varyans korunumu, SELU sabit noktası |
| M3.6 normalizasyon | 11 | 3 | 20 katmanda normalizasyon (aktivasyon std'si ve gradyan normu), BN/LN/IN/GN/RMSNorm eksen gezgini (+ mevcut BN gürültüsü) | ölçek değişmezliği ve etkin öğrenme oranı, BN geri geçişi, BN katlama |
| M3.7 düzenlileştirme | 9 | 2 | dropout + weight decay + early stopping + MC dropout bandı, gradyan kırpma (değer vs norm) | early stopping ≈ L2 (α ≈ 1/ηt), inverted dropout momentleri, doğrusal modelde dropout = veriye uyarlanmış L2, alpha dropout afin düzeltmesi |
| M3.8 perceptron | 13 | 3 | perceptron + Novikoff sınırı + öznitelik dönüşümü, ReLU menteşeleri ve çadır haritası, çift iniş | Novikoff kanıtı, XOR'un ayrılamazlığı, perceptron = perceptron kriterinin SGD'si, ara değerleme hatası max|f″|/(8K²), derinlikle 2ᴸ parça |
| M3.9 eğitim pratiği | 9 | 2 | öğrenme ve kapasite eğrileri, veri sızıntısı (grup, öznitelik seçimi), sınıf ağırlıkları | ağırlıklı CE'nin logit'i log w kaydırması |

- **30 yeni etkileşimli deney** (toplam 71). Her birinin iddiaları birkaç tohumla sayısal olarak doğrulandıktan sonra yazıldı; doğrulanamayan iddialar metinden çıkarıldı (aşağıda).
- **Deney sonucuna göre düzeltilen anlatımlar:**
  - M2.2: "az veride Naive Bayes lojistik regresyondan iyidir" iddiası 2-B deneyde görünmüyordu. İddia kaldırıldı; yerine kalibrasyon grafiği ve iddianın gerçekten geçerli olduğu yüksek boyutlu bir deney eklendi.
  - M3.6: normalizasyon deneyinin ilk notu "gradyanlar katmanlar boyunca dengelenir" diyordu. Ölçüm, artıksız derin BN ağında ilk katmanların gradyanının 20 katmanda ≈ 40×, 40 katmanda binlerce kat büyük olduğunu gösterdi (Yang ve ark. 2019 ile uyumlu); not ve anlatım buna göre düzeltildi.
  - M3.9: gruplu teşhis eşiği tek tohumda yanıltıcı etiketler veriyordu; σ²/2 eşikli kurala geçildi ve deneyin sınırlarına yazıldı.
- **Ders slaytlarındaki hatalar (anlatımda kutuyla belirtildi):**
  - BN katlama formülü σ + ε değil √(σ² + ε) olmalı; sabit bir kanalda iki ifade 300 kat ayrışır.
  - 200×200 RGB görüntü sayısı 40 000 × 256³ değil (256³)^40 000 = 2^960 000.
  - 138 milyon parametre VGG-19'un değil VGG-16'nın; VGG-19 ≈ 144 milyon.
  - MLE "tüm tutarlı kestiriciler arasında en düşük MSE" değil, asimptotik olarak verimlidir.
  - PReLU'nun eğimi hiperparametre değil, öğrenilen bir parametredir.
  - "Geri geçiş = güncelleme adımı" değil: geri geçiş gradyanları hesaplar, güncellemeyi optimizer yapar.
- **Diğer düzeltmeler:**
  - M2.6: sıkıştırma oranı ≈ %67 olarak düzeltildi.
  - M2.4: "kesin içbükey" ifadesi düzeltildi.
  - M3.1: ifade ayrıştırıcısının yanıltıcı hata mesajı ("'*' tanımsız") düzeltildi.
  - M3.4: log eksende çubuklar görünmüyordu.
  - M3.2 ve M3.5: artık bağlantılara (ResNet) yapılan çapraz başvuru yanlış modüle (M4.8) gidiyordu; M4.2'ye çevrildi.
  - SVM, k-means Voronoi ve EM alt sınır şekilleri kesin hesapla yeniden çizildi.
  - Grafiklerde sağ kenara yakın işaret etiketleri kesiliyordu; artık sola dönüyor.
- **Kalan:** Faz 4–11 konu anlatımları (sırayla); Faz 5–11'in kalan modüllerine pratik kartları; kapsam tablosunun güncellenmesi.

## 2026-11 (3. tur) · Tam konu anlatımları (Faz 0 ve Faz 1)

Geri bildirim: modül girişleri kısa bir özetten ibaretti. Her modüle artık aşamalardan **önce** gelen, slayt düzeyinde ya da daha ayrıntılı bir **Konu anlatımı** bölümü ekleniyor (sayfadaki "§ Konu anlatımı", içindekiler ve okuma süresiyle). Her anlatım aynı düzeni izler: problem → sezgi → kesin tanım → türetim/kanıt (koşullarıyla, açılır kutuda) → sayısal örnek → şekil → etkileşimli deney → yanılgılar → PyTorch karşılığı → özet tablosu → kaynaklar.

| modül | bölüm | yeni şekil | deneyler (yeni) | öne çıkan türetimler |
|---|---|---|---|---|
| M0.1 iç çarpım | 12 | 5 | iki vektörü sürükle, Lp birim topları, yüksek boyutta kosinüs | kosinüs teoremi, izdüşüm artığının dikliği, Cauchy–Schwarz, üçgen eşitsizliği |
| M0.2 matris çarpımı | 13 | 2 | dönüşüm ızgarası (sütunları sürükle), döngü sırası hız ölçümü | bileşke = çarpım, birleşme ve (AB)ᵀ = BᵀAᵀ kanıtı, dA = G Bᵀ ve dB = Aᵀ G |
| M0.3 türev ve zincir kuralı | 13 | 3 | sekant/teğet ve sayısal türev hatası, gradyan ve kontur, adım adım geri yayılım, Jacobian'ın yerel yaklaşımı | sigmoid türevi, yönlü türev ve en dik çıkış, softmax Jacobian'ı, zincir kuralı kanıtı, ileri/merkezi fark hata analizi |
| M0.4 olasılık ve MLE | 15 | 2 | Bayes ızgarası (1000 kişi), dağılımdan örnekleme, merkezi limit teoremi, olabilirlik/MAP | σ/√n, Bernoulli/Gauss/kategorik MLE, MLE varyansının yanlılığı, MSE/BCE/CE = NLL, NLL = H + KL |
| M0.5 tensör araçları | 10 | 1 | storage ve stride görselleştirici, zaman ölçümü istatistiği | bitişik stride formülü, kayan pencere stride'ı |
| M1.1 doğrusal regresyon | 10 | 1 | kapalı form vs GD ve koşul sayısı, dayanıklı kayıplar | MSE gradyanı, normal denklemler, izdüşümün en yakın nokta olduğu (Pisagor), η < 2/λ_max ve (κ−1)/(κ+1) hızı |
| M1.2 GD varyantları | 6 | 1 | mini-batch gürültü topu (+ mevcut 1-B GD, 2-B momentum) | gürültü topu varyansı η s²/(B a (2 − ηa)), momentumun √κ hızlanması |
| M1.3 düzenlileştirme | 8 | 1 | ridge/lasso yolları ve doğrulama hatası, L1/L2 geometrisi (+ mevcut polinom) | yanlılık–varyans ayrışımı, ridge SVD küçültmesi, soft-threshold, KKT, weight decay eşdeğerliği |
| M1.4 lojistik regresyon | 8 | 1 | 2-B olasılık haritası ve ‖w‖, kayıplar ve sayısal kararlılık | ∂ℓ/∂z = p − y, BCE'nin dışbükeyliği, softplus ile kararlı kayıp |
| M1.5 softmax | 8 | 0 | 3 sınıflı karar bölgeleri (+ mevcut sıcaklık) | ∂ℓ/∂z = p − onehot, log-sum-exp hilesi |

- **23 yeni etkileşimli deney** (toplam 41). Sürüklenebilir şekiller (vektörler, matris sütunları, Jacobian noktası) fare, dokunma ve klavye (ok tuşları) ile çalışır; her birinin hesapları otomatik testlerle doğrulanır (`selfTest`). Standart deneylerde yine her kontrolün sonucu gerçekten değiştirdiği, geçersiz değerlerin açıklamayla reddedildiği ve aynı tohumun aynı sonucu verdiği test edilir.
- **Düzeltilen hatalar:** M1.2 "Aynı η, iki eğrilik" senaryo notu kendisiyle çelişiyordu ("ıraksar" deyip ηa = 1.8 < 2 diyordu) — düzeltildi. Ders tabloları ve uzun satır içi kodlar dar ekranda sayfayı yatay taşırıyordu — tablolar kendi içinde kayıyor, kod kırılıyor. Sürükleme sırasında metin seçimi şekillerin üstünde mavi kutular bırakıyordu — kapatıldı.
- **CI:** üç modülün (M4.10, M4.14, M5.6) eşikleri CI donanımındaki gürültüye göre gevşetildi; site yeniden yayında.
- **Kalan:** Faz 2–11'in konu anlatımları (sırayla, aynı düzende); Faz 5–11'in kalan modüllerine pratik kartları.

## 2026-11 (2. tur) · Faz 0, 2, 3 ve 4'te pratik

- **Faz 0** (iç çarpım, matris çarpımı, olasılık/MLE, tensör belleği), **Faz 2** (klasik ML, 8 modül), **Faz 3'ün kalanı** (micrograd, tensör autograd, perceptron, eğitim pratiği) ve **Faz 4'ün kalan 13 modülü** (CNN, LSTM/GRU, BPE, Transformer, CNN blokları, tespit, segmentasyon, zaman serisi, word2vec, seq2seq, kodlayıcı–kod çözücü Transformer, ViT, ön eğitimli dil modelleri) pratik kartları, sık yanılgılar ve "koşul değişirse" sorularıyla genişletildi. **Faz 0–4 artık bütünüyle kapsandı.**
- Toplam: **46 modülde 252 kart** (46 çözülmüş örnek, 92 temel, 70 uygulama, 44 zorlayıcı); türler: hesaplama 98, yöntem seçme 23, açıklama 18, karşı örnek 16, yorumlama 14, hata bulma 12, kanıt 11, kod 4, birleştirme 4, uyarlama 3, eksik adım 2, proje 1. **111 yanılgı, 71 koşul sorusu.**
- Yeni çözülmüş türetim ve kanıtlar: Cauchy–Schwarz, matris zinciri çarpım sırası, Bernoulli MLE, stride ile adresleme, micrograd geri geçişi, yayının gradyanı (unbroadcast), Novikoff yakınsama teoremi, XOR'un ayrılamazlığı, uzaklık matrisi açılımı ve sayısal yok etme riski, Gini/entropi kazancı, bagging varyansı $\rho\sigma^2+(1-\rho)\sigma^2/B$, pseudo-residual = negatif gradyan, ortak varyanslı Gaussian NB'nin doğrusal oluşu, polinom çekirdeğinin öznitelik haritası, Lloyd'un sonluluğu, EM'in sert limiti olarak k-means, AR(1) momentleri, sinüzoidal konum kodlamasının göreli konumu dönme olarak taşıması, transposed conv çıkış boyu, Transformer bloğunun 12d² + 13d parametresi, DPO'nun RLHF'den türetilmesi.
- Kontrol: tüm sayısal cevaplar bağımsız hesapla doğrulandı; sık hata değerlerinin doğru cevaptan farklı olduğu test edildi (iki çakışma yakalanıp düzeltildi); çözülmüş örneklerde tekrar eden etiket kaldırıldı.

## 2026-11 (1. tur)

Bu sayfa, kaynak ekleme kurallarına göre yapılan geliştirmeleri denetlenebilir biçimde listeler: ne eklendi, hangi kaynaktaki eksik kapatıldı, neler düzeltildi ve neler henüz eksik. Hangi kaynak bölümünün hangi sayfaya eşlendiği [Kapsam tablosu](kapsam.html)'ndadır.

## 2026-11 · Öğrenme deneyimi ve kapsam denetimi

### Kapsam denetimi (ders notları arşivi)

82 PDF'in metni çıkarıldı; her slayt başlığı ilgili modüllerin tüm dosyalarında arandı, eşleşmeyenler elle incelendi. Önceki genişletmede kapsandığı varsayılan ama gerçekte eksik kalan konular bulundu ve tamamlandı:

| Kaynak bölümü | Eksik | Eklenen |
|---|---|---|
| Paralel, ders 3 (thread'ler) | semaforlar, monitörler, CAS ve LL/SC, ABA, livelock, sonlandırma (zehirli hap), call_once, sıralı tutarlılık / doğrusallaştırılabilirlik, fiber'ler, hata ayıklama | M10.2: ek alıştırma E4 (semaforlu sınırlı tampon, rastgele zamanlayıcıyla kilitlenme ve sonlandırma testi), E5 (Treiber yığını, ABA'nın yalın CAS'ta görülüp etiketli CAS'ta görülmemesi), üç ayrıntı kutusu |
| Görüntü işleme, ders 1–2 | görüntüleme modaliteleri ve EM spektrum, Bayer deseni | M9.1: sezgi ve tanım kutuları, ek alıştırma E4 (mozaik + çift doğrusal demozaikleme, kanal başına PSNR, kenarda renk hatası) |
| Görüntü işleme, ders 10–11 | metamerizm, trikromatik katsayılar | M9.9: tanım kutusu |
| Görüntü işleme, ders 23 | sabitleme zorlukları ve yöntem aileleri | M9.17: ayrıntı kutusu |
| Paralel, ders 6 | hizalama, doku belleği | M10.4: ayrıntı kutusu (uygulama GPU gerektirdiği için kavramsal) |
| RL, ders 1 ve 6 | öğrenme türleri ve uygulamalar; ajan türleri, keşif yöntemleri | M11.1 sezgi kutusu; M11.4 tanım kutusu + ek alıştırma E4 (ε-greedy, azalan ε, Boltzmann, iyimser başlangıç zincir ortamında) |
| Derin öğrenme, bölüm 14 | görsel korteks esini | M4.1 sezgi kutusu |

### Yeni öğrenme altyapısı

- **Katmanlı anlatım kutuları** (`:::türetim`, `:::kanıt`, `:::sezgi`, `:::tanım`, `:::uyarı`, `:::yanılgı`, `:::koşul`): türetim ve kanıtlar açılır panellerde, sezgi ve tanımlar metnin içinde. Yeni türetimler: PCA (Lagrange ile en büyük varyans yönü), SVM marjı ve hinge kaybının kaynağı, EM'in monotonluğu, LSTM hücre yolu gradyanı, GAN en iyi ayırıcı → Jensen–Shannon (adım adım, varsayımları ve neden pratikte bozulduğu), Naive Bayes ve Laplace düzeltmesinin Bayesçi yorumu.
- **Pratik kartları:** 17 modülde 95 kart (17 çözülmüş örnek, 34 temel, 28 uygulama, 16 zorlayıcı); türler: hesaplama 36, açıklama 6, yorumlama 6, yöntem seçme 6, kanıt 5, karşı örnek 5, hata bulma 4, uyarlama 3, birleştirme 3, eksik adım 2, kod 2. Her kartta hedef, gerekenler, zorluk, kademeli ipuçları ve başta gizli açıklamalı çözüm; uygun olanlarda ikinci çözüm yolu. Sayısal cevaplar test sırasında `calc` ifadesiyle yeniden hesaplanır; sık yapılan hatalar tanınır ve nedeni söylenir (ör. "ortalamaları çıkarmadan Σxy/Σx² kullandın"). Açık uçlu kartlarda otomatik puan yok: değerlendirme ölçütleri ve örnek çözüm var.
- **Sık yanılgılar ve "koşul değişirse"** bölümleri: aynı 17 modülde 50 yanılgı, 34 koşul sorusu.
- **El hesabı:** 100 modülde çözüm yolu artık başta gizli (yalnız soruyu kuran açıklamalar görünür). Alanlara birden çok kademeli ipucu ve sık hata mesajı eklenebiliyor; işaret ve basamak kaymaları her alanda otomatik tanınıyor.
- **Kod aşamaları:** kademeli ipucu desteği (`hints:`).
- **Etkileşimli deneyler:** 18 deney, 19 yerde: GD kararlılığı, optimizasyon yöntemleri (2B kötü koşullu vadi), EMA, başlatma × aktivasyon, polinom regresyonu ve ridge, softmax sıcaklığı (iki modülde), BN batch gürültüsü, 1B konvolüsyon, RNN gradyan akışı, √d_k ölçekleme, gürültü takvimi, CFG ölçeği, k-means, değer iterasyonu, haydutlar, yoğunluk dönüşümleri, bilateral süzgeç, Amdahl/Gustafson. Her birinde kaydırıcı + kesin değer, başlangıca dön, açıklamalı hazır senaryolar, A/B karşılaştırma, önce-tahmin-et sorusu, tohum, geçersiz değer açıklamaları, "deneyle cevapla" soruları ve modelin sınırları. Hesaplar gerçektir; bir ayar seçili modda kullanılmıyorsa pasifleşir.
- **Önkoşul haritası** (harita.html) ve önkoşullara saygılı önerilen öğrenme sırası (metin sürümü de var).
- **Başlangıç yoklaması:** her modülün başında önkoşul modüllerinin sınama sorularından "Hazır mısın?".
- **Öz değerlendirme:** kazanımlar işaretlenebilir; sayfa açmak ilerleme sayılmaz. Modül sonunda "geri dönmen önerilen yerler" (ilk denemede yanlış sorular, işaretlenmemiş kazanımlar, geçilmemiş aşamalar).
- **Karışık tekrar** (tekrar.html): tüm modüllerin sınama soruları üzerinde Leitner kutuları.
- **Notlar, yer imleri, kaldığın yer** (notlar.html; dışa/içe aktarma).
- **Site içi arama** (başlıkta, `/` kısayolu, klavyeyle gezinme).

### Düzeltmeler

- Mobilde kapalı sözlük ipuçları sayfayı yatayda taşırıyordu (görünmez ama yer kaplayan ipucu kutuları); kapalıyken yerleşimden çıkarıldı, mobilde ekranın altında açılıyor.
- Değerlendirme fieldset'leri dar ekranda taşıyordu; düzeltildi.
- Üst menü dar masaüstü ve mobilde taşmasın diye bağlantılar yan menüye taşınıyor.

### Yapılan kontroller

Model çözümlerin tüm kontrolleri geçmesi ve iskeletlerin geçmemesi (101 modül); pratik kartlarının sayısal cevaplarının bağımsız hesapla eşleşmesi, tek doğru seçenek, ipucu ve çözüm varlığı; her deneyin varsayılanla ve tüm hazır senaryolarla çalışması, **her kontrolün sonucu gerçekten değiştirmesi**, aralık dışı değerlerin açıklamalı reddi, aynı tohumla tekrarlanabilirlik, konvolüsyon çıktı boyu formülünün 300 rastgele ayarda sayılan değerle eşitliği; değiştirilen modüllerde tarayıcıda uçtan uca çalıştırma; masaüstü ve iPhone boyutunda ekran görüntüsü ve yatay taşma denetimi; iç bağlantılar.

### Henüz eksik olanlar

- (1. tur sonunda) Pratik kartları 17 modüldeydi; 2. turda 46'ya çıktı (Faz 0–4 tamam). Kalanlar: Faz 5 (VAE ve difüzyon dışında), Faz 6–11 (MDP dışında): 55 modül.
- Statik açıklayıcı diyagramlar (hesap grafiği, mimari çizimleri) henüz yok; görseller deneylerin gerçek hesaplı grafikleriyle sınırlı.
- Slaytlardaki şekiller metin olarak okunamadığı için karşılaştırma başlık ve metin düzeyinde yapıldı.
- Doku belleği (M10.4) ve CUDA alıştırmaları tarayıcıda GPU olmadığı için kavramsal/Colab'da.
