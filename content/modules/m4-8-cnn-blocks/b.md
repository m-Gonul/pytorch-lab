Saf Python sayaçları. Katman tanımları sözlük: `{"type": "conv", "k": 3, "cin": 16, "cout": 32, "groups": 1, "bias": True}`, `{"type": "pool", "k": 2}`, `{"type": "linear", "in": 576, "out": 64}`, `{"type": "bn", "c": 64}`, `{"type": "flatten"}`.

- `count(layers, in_shape)` → `(parametre, MAC, çıkış şekli)`; `in_shape = (C, H, W)`; konv. "same" dolgu, adım 1; havuz adım = k (taban); BN 2C eğitilebilir parametre; MAC: konv. $H W\,K^2C_{in}C_{out}/g$, doğrusal in·out.
- `inception_params(cin, c1, c3r, c3, c5r, c5, pp)` — GoogLeNet modülü (bias yok): 1×1 dalı c1; 1×1 c3r → 3×3 c3; 1×1 c5r → 5×5 c5; 3×3 havuz → 1×1 pp. Çıkış kanal sayısı ile birlikte.
- `lrn(a, k=2, n=5, alpha=1e-4, beta=0.75)` — `a` kanal listesi (tek piksel), AlexNet LRN (pencere n, α bölünmeden).
