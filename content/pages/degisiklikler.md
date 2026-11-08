# Değişiklik günlüğü

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
