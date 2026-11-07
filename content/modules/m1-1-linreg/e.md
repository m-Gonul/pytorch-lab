Hangi learning rate? Tipik cevap: birkaç değer dene. Döngüyle 64 lr × 200 adım = 12 800 küçük adım; her biri birkaç kernel çağrısı.

Bunun yerine 64 modeli **tek model** gibi eğit: ağırlıkları `W: (L, d)`, `b: (L,)` olarak tut. Tahmin `X @ W.T + b` → `(n, L)`: her sütun bir model. Kayıplar `(L,)`; `loss.sum().backward()` her modelin kendi gradyanını verir (sınama sorusuna bak). Güncellemede her model kendi lr'si ile ölçeklenir: `W -= lrs[:, None] * W.grad`.

- `sweep(X, y, lrs, steps=200)` → `(W, b, losses)`; `losses` son kayıplar `(L,)`.

Kontrol yalnız **adım döngüsüne** izin verir; lr üzerinde döngü yok. Sonuç, her lr ile tek tek eğitilmiş modellerle karşılaştırılır ve hız tablosu üretilir. Iraksayan (çok büyük lr) modellerin kaybı `inf`/`nan` olabilir; bu beklenen bir sonuç.
