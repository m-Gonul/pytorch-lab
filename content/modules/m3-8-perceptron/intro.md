**Biyolojik ilham:** nöron, dendritlerden gelen sinyalleri toplar, eşik aşılırsa akson boyunca ateşler. McCulloch & Pitts (1943): ikili girdi/çıktılı yapay nöronla VE, VEYA, DEĞİL hesaplanır; ağları her mantıksal önermeyi hesaplar.

**TLU** (eşik mantık birimi): $z=w^\top x+b$, $\hat y=\mathrm{step}(z)$ — doğrusal bir karar sınırı. **Perceptron** (Rosenblatt 1957): TLU'lardan oluşan tek katman; öğrenme kuralı her hatalı örnekte $w\leftarrow w+\eta(y-\hat y)x$ ("birlikte ateşleyen birlikte bağlanır"ın hataya dayalı biçimi). **Yakınsama teoremi:** veri bir γ kenar payıyla doğrusal ayrılabiliyorsa ve $\|x\|\le R$ ise hata sayısı $\le(R/\gamma)^2$. Ayrılabilir değilse dolaşır (XOR).

**Doğrusal sınıflandırıcı + dönüşüm:** 2B halkalar doğrusal ayrılamaz; $(x_1,x_2)\mapsto(x_1,x_2,x_1^2+x_2^2)$ ile 3B'de bir düzlem ayırır (çekirdek yönteminin fikri). **MLP** aynı nöron modelini yığar (ağırlıklı toplam + doğrusalsızlık, tekrar tekrar): son katman, gizli katmanların **öğrendiği** dönüşüm üzerinde bir doğrusal sınıflandırıcıdır.

**Evrensel yaklaşım** (Cybenko 1989, Hornik 1991): tek gizli katman, yeterli genişlikle kompakt bir kümedeki her sürekli fonksiyonu yaklaşıklar. 1B'de k kırılma noktalı parçalı doğrusal fonksiyon k ReLU ile tam ifade edilir. Ama genişlik üstel büyüyebilir; **derinlik** bileşke yoluyla parçaları üstel çoğaltır (testere dişi örneği). Ders alıştırması: `MLPRegressor` ile katman sayısını artırarak skaler (1B), 2B Gauss $e^{-(x_1^2+x_2^2)/0.1}$ ve iki çıktılı vektör fonksiyonu yaklaşıklamak.

**Çift iniş:** klasik görüş — kapasite arttıkça test hatası U çizer. Rastgele özniteliklerle en küçük normlu çözümde, parametre sayısı p veri sayısı n'ye yaklaşınca hata patlar (enterpolasyon eşiği), p ≫ n'de yeniden düşer ve çoğu zaman U'nun dibinin altına iner.
