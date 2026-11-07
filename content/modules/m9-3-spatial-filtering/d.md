- `filter2d(img, k, border)` — PyTorch'ta korelasyon (`F.conv2d`); `border` OpenCV adıyla: `"constant"`, `"replicate"`, `"reflect101"` → `F.pad` modları `constant`, `replicate`, `reflect`.
- `recover_kernel(x, y, ksize, steps=500, lr=0.1)` — `y = filter2d(x, w*, "reflect101")` olan bilinmeyen `w*`'ı, `w` parametresini `Adam` ile MSE'yi küçülterek bul. Bir CNN katmanının "öğrendiği" şey tam olarak budur.

Deney: aynı şeyi doğal bir görüntüyle (`camera`) dene. Kayıp neredeyse sıfıra iner ama bulunan çekirdek gerçeğinden çok farklıdır: komşu pikseller birbirine çok benzediği için normal denklemler kötü koşullanmıştır, birçok çekirdek neredeyse aynı çıktıyı verir. Sistem tanımada bu yüzden **beyaz gürültü** girdisi kullanılır; kontrol de onu kullanıyor.
