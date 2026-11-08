# Değişiklik günlüğü

## 2026-11 (2. tur) · Faz 2 ve Faz 4'te pratik

- **Faz 2 (klasik ML, 8 modül)** ve **Faz 4'ün kalan 13 modülü** (CNN, LSTM/GRU, BPE, Transformer, CNN blokları, tespit, segmentasyon, zaman serisi, word2vec, seq2seq, kodlayıcı–kod çözücü Transformer, ViT, ön eğitimli dil modelleri) pratik kartları, sık yanılgılar ve "koşul değişirse" sorularıyla genişletildi.
- Toplam: **38 modülde 212 kart** (38 çözülmüş örnek, 76 temel, 61 uygulama, 37 zorlayıcı); türler: hesaplama 81, yöntem seçme 22, açıklama 15, karşı örnek 14, yorumlama 12, kanıt 10, hata bulma 8, birleştirme 4, uyarlama 3, kod 3, eksik adım 2. **95 yanılgı, 63 koşul sorusu.**
- Yeni çözülmüş türetimler: uzaklık matrisi açılımı ve sayısal yok etme riski, Gini/entropi kazancı, bagging varyansı $\rho\sigma^2+(1-\rho)\sigma^2/B$, pseudo-residual = negatif gradyan, ortak varyanslı Gaussian NB'nin doğrusal oluşu, polinom çekirdeğinin öznitelik haritası, Lloyd'un sonluluğu, EM'in sert limiti olarak k-means, AR(1) momentleri, sinüzoidal konum kodlamasının göreli konumu dönme olarak taşıması, transposed conv çıkış boyu, Transformer bloğunun 12d² + 13d parametresi, DPO'nun RLHF'den türetilmesi.
- Kontrol: tüm sayısal cevaplar bağımsız hesapla doğrulandı; sık hata değerlerinin doğru cevaptan farklı olduğu test edildi (bir çakışma yakalanıp düzeltildi).

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

- (1. tur sonunda) Pratik kartları 17 modüldeydi; 2. turda 38'e çıktı. Kalanlar: Faz 0'ın diğer modülleri, Faz 3'ün kalanı, Faz 5–11 (difüzyon/VAE dışında).
- Statik açıklayıcı diyagramlar (hesap grafiği, mimari çizimleri) henüz yok; görseller deneylerin gerçek hesaplı grafikleriyle sınırlı.
- Slaytlardaki şekiller metin olarak okunamadığı için karşılaştırma başlık ve metin düzeyinde yapıldı.
- Doku belleği (M10.4) ve CUDA alıştırmaları tarayıcıda GPU olmadığı için kavramsal/Colab'da.
