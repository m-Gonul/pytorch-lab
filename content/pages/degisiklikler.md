# Değişiklik günlüğü

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
