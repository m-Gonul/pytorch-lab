<!-- çözüm -->

Üç örneği normalize et; sonra yalnız ilk çıktıya gradyan gönder ($dy=[1,0,0]$). Gradyan yalnız $x_1$'e değil, **üç** girdiye de dağılıyor ve toplamı sıfır: $0.408-0.816+0.408=0$. Tüm girdileri aynı miktarda kaydırmak $\hat x$'yi değiştirmez; gradyan da bu yönde sıfır.

RMSNorm'da ortalama çıkarılmıyor; yalnız ölçek normalize ediliyor. Daha ucuz ve LLaMA gibi modellerde LayerNorm kadar iyi çalışıyor.
