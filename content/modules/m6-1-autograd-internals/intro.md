PyTorch grafı **çalışırken** kurar: `requires_grad` bir tensörü kullanan her işlem, çıktıya bir `grad_fn` düğümü bağlar. Düğümler `next_functions` ile girdilerinin düğümlerine işaret eder; yaprak tensörlerin düğümü `AccumulateGrad`'dir ve gradyanı `.grad`'a **ekler**.

`backward()`:

1. Çıktıdan başlayıp grafı topolojik sırayla gezer (her düğüm, tüm tüketicileri işlendikten sonra).
2. Her düğümde gelen gradyanları toplar, hook varsa uygular, kendi türevini çarpıp girdilere dağıtır.
3. Yapraklarda `.grad += …`.

Türevi hesaplamak için düğümler bazı tensörleri **saklar** ([[saved tensors]]): `mul` iki girdisini, `exp` ve `tanh` çıktılarını, `matmul` iki girdisini. Eğitimdeki aktivasyon belleğinin büyük kısmı bunlardır.

Saklanan bir tensör sonradan in-place değiştirilirse geri geçiş yanlış olur. PyTorch bunu [[version counter]] ile yakalar: her in-place işlem tensörün sürümünü artırır; düğüm sakladığı sürümü geri geçişte karşılaştırır.
