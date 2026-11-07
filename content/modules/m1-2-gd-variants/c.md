NumPy'da çok özellikli regresyon. Eğitim döngüsü iki katmanlı: epoch'lar ve her epoch'taki batch'ler. Örnekler **üzerinde** döngü yok (kontrol `train` içinde en fazla 2 katman döngüye izin verir).

- `train(X, y, lr, batch_size, epochs, beta=0.0, seed=0)` → `(w, b, losses)`. Her epoch başında `rng = np.random.default_rng(seed)` ile oluşturduğun üreteçten `rng.permutation(n)`. `losses[k]`: $k$. epoch **sonundaki** tam veri MSE'si. Momentum `w` ve `b` için ayrı hız vektörleriyle.

Kontrol, kötü koşullu bir problemde (`X`'in sütun ölçekleri 1 ve 10) aynı lr ile $\beta=0.9$'un $\beta=0$'dan belirgin şekilde daha düşük kayba indiğini doğrular.
