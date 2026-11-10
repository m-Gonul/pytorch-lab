# Değişiklik günlüğü

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
