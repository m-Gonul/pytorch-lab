Softmax'ı elle hesapla, sonra kaybı iki yoldan kontrol et: $-\ln p_0$ ile $\text{LSE}(z)-z_0$ aynı sayıyı vermeli.

<!-- çözüm -->

Gradyan vektörü $p-e_y=[-0.335,\ 0.245,\ 0.090]$: bileşenleri toplamı **sıfır**. Gradient descent doğru sınıfın logit'ini yükseltir, diğerlerini kendi olasılıkları oranında düşürür.

Son grupta $z'=z+998$: softmax değişmez. Bilgisayarda $e^{1000}$'i doğrudan hesaplamaya kalkmak `inf/inf = nan` verir.
