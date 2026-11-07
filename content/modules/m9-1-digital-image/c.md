NumPy, **döngü yok**.

- `resize_bilinear(img, H2, W2)` — her eksen için kaynak indeksleri ve ağırlıklar vektörü, sonra gelişmiş indeksleme.
- `resize_cubic(img, H2, W2, a=-0.75)` — ayrılabilir: önce satırlarda 4 komşu ağırlıklı toplam, sonra sütunlarda. İndeksler kenarda kırpılır (`BORDER_REPLICATE`).

Kontrol tam boyutlu `camera` görüntüsünde büyütme ve küçültme için `cv2.resize` ile karşılaştırır (tolerans 0.01 gri seviye: tarayıcıdaki WebAssembly OpenCV ağırlıkları biraz farklı yuvarlıyor).
