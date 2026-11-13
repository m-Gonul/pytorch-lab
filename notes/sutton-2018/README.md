# Sutton ve Barto (2018): üretim kaydı

Bu kaynak tamamlanmadı. Birim defteri `content/coverage/sutton.yaml` içindedir. `progress.complete: false` kitabın tamamına ilişkindir. İlk işlenen aralık basılı s. 1–22, PDF s. 23–44.

Bölüm 1 şekillerini yeniden hesaplamak için (yalnız Python standart kitaplığı):

```bash
python3 -I notes/sutton-2018/generate_figures.py /tmp/sutton-ch1
```

Çıktılar `board.html`, `tree.html`, `ch1-trace.json`. Tahta kazancı ve yörüngedeki c/g açgözlü seçimleri, e için daha iyi seçilmeyen alternatif bulunması, tam oyun ağacı üzerinde denetlenir. X optimal, O düzgün rastgele varsayılır. Bu, kaynak görselin yasal hamle ve öğrenme ilişkilerini taşıyan özgün bir örnek yörüngedir; kaynakta verilmeyen sayılar ona atfedilmez.

Ek `explorevalue` deneyi farklı bir iki aşamalı ağaçtır; kaynak Alıştırma 1.4 ayrımını açıklamak içindir. XOX eğitim grafiği değildir.

İlk envanterin etiketli sayıları 140 alıştırma, 104 şekil, 38 gri kutulu sözde-kod ve 1 numaralı tablodur. Numarasız görseller ve metin içi algoritmaların bütün kaynak sayımı henüz bitmedi. İşlenmeyen hazır modüller, izin verilen anlamı dışında `bekliyor` olarak işaretlenmedi.
