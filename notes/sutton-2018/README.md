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
