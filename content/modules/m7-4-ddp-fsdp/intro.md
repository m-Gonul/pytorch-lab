**Veri paralelliği:** her işçi (GPU/süreç) modelin tam bir kopyasını tutar, batch'in farklı bir parçasını işler. Gradyanlar işçiler arasında **ortalanır** ([[all-reduce]]); herkes aynı güncellemeyi yapar ve kopyalar senkron kalır. Ortalama kayıpla bu, tüm parçaların birleşimi üzerindeki tek bir büyük batch adımına **eşittir**.

**Ring all-reduce:** vektörü $N$ parçaya böl; işçiler halka hâlinde komşularına parça gönderir.

1. *Reduce-scatter* ($N-1$ adım): her adımda alınan parça kendi parçana eklenir; sonunda her işçi bir parçanın tam toplamına sahip.
2. *All-gather* ($N-1$ adım): toplanmış parçalar halka boyunca dağıtılır.

İşçi başına gönderilen: $2(N-1)/N\cdot$ boyut.

**Bellek:** DDP'de her işçi parametre + gradyan + optimizer durumunun tamamını tutar. **[[ZeRO]]** bunları böler: ZeRO-1 optimizer durumunu, ZeRO-2 + gradyanları, ZeRO-3 (= FSDP) + parametreleri. ZeRO-1'de her işçi yalnız kendi parametre parçasını günceller, sonra parçalar all-gather ile birleştirilir.

`torch.distributed` CPU'da `gloo` arka ucuyla çalışır: bu modülde iki süreç aynı makinede `127.0.0.1` üzerinden konuşur.
