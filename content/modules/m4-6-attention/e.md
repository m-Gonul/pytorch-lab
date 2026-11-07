`blocked_attention(q, k, v, block=64, causal=False)` — key/value'ları `block` uzunluğunda parçalar hâlinde gez ve [[online softmax]] ile birleştir. **Yalnız blok döngüsü** (tek katman). Satır başına `m` (koşan max, başlangıç $-\infty$), `l` (payda), `acc` (pay).

Dikkat: causal'da bir satırın bir bloktaki tüm skorları $-\infty$ olabilir, hatta o ana kadarki `m` de $-\infty$ olabilir; $-\infty-(-\infty)$ `nan` verir. `m_new` $-\infty$ ise çıkarılacak değer olarak 0 kullan.

Kontrol sonucu ve gradyanı `F.scaled_dot_product_attention` ile karşılaştırır. Asıl kazanç bellekte: tam skor matrisi yerine `T × block`. CPU'da da blok önbelleğe sığdığı için genelde hızlanır; GPU'da bu fikir SRAM/HBM farkından büyük hız getirir.
