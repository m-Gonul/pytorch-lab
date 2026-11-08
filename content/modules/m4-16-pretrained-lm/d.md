PyTorch. Sentetik dil: `corpus(n, seed)` verilir — `[CLS]` + 16 token; çift konumlarda "işlev" kelimeleri (bir sonraki içerik kelimesinin belirlenimci fonksiyonu), tek konumlarda içerik kelimeleri (cümlenin konusuna ait 8 kelimeden ya da %50 olasılıkla tüm konulardan rastgele). Aşağı akış görevi: konu (5 sınıf).

- `mlm_mask(x, g, V, p=0.15)` — torch sürümü (özel token'lar 0–3: PAD, CLS, SEP, MASK).
- `TinyBERT(V, d=64, L=2)` — embedding + öğrenilebilir konum + Pre-LN kodlayıcı + LN; `mlm` başı (d → V) ve `cls` başı (d → 5); `forward(x)` gizli durumlar.
- `pretrain(model, U, epochs, lr, seed)` — MLM kaybı (yalnız seçilen konumlar).
- `finetune(model, X, y, steps, lr, freeze=False)` — `[CLS]` durumu → `cls` başı; `freeze=True`: yalnız baş (doğrusal sonda).
