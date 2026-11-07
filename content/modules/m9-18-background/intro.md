**Arka plan çıkarma:** ön plan (hareket eden, ilgi çeken) ile arka plan piksellerini ayırmak; gözetim, trafik, hareket tabanlı anomali tespiti için ön işleme bloğu. Zorluklar: gürültü, aydınlatma değişimi, dalgalanan arka plan (yaprak, su), kamera titreşimi, gölgeler, duran nesneler ve **hayaletler** (ilk karede bulunup sonra giden nesnenin bıraktığı yanlış ön plan).

**Basit modeller:** kare farkı $|I_t-I_{t-1}|>T$ (iç kısımları kaçırır, iki kenar verir); üç kare farkı; **üstel hareketli ortalama** $B_t=(1-\alpha)B_{t-1}+\alpha I_t$, ön plan $|I_t-B_{t-1}|>k\sigma$; zamansal medyan.

**Stauffer–Grimson GMM:** her piksel $K$ (3–5) Gauss'luk karışım. Yeni değer, sırayla ($w/\sigma$ büyükten küçüğe) $|x-\mu_k|<2.5\sigma_k$ olan ilk bileşenle eşleşir; eşleşenin ağırlığı artar, ortalama ve varyansı güncellenir; eşleşme yoksa en zayıf bileşen $x$ merkezli geniş bir bileşenle değiştirilir. Arka plan: sıralamada birikimli ağırlığı $T$'yi geçene kadarki bileşenler. OpenCV MOG2 bunun $K$'yı uyarlayan sürümüdür.

**ViBe** (Barnich & Van Droogenbroeck): **parametresiz**, piksel başına geçmiş değerlerden $N\approx20$ örnek.

- *Başlatma:* tek kare yeter — her örnek pikselin rastgele bir 8-komşusundan.
- *Sınıflandırma:* yarıçap $R\approx20$ içinde en az $\#_{min}=2$ örnek varsa arka plan.
- *Güncelleme (yalnız arka plan pikselleri):* $1/\phi$ ($\phi=16$) olasılıkla **rastgele** bir örneği şimdiki değerle değiştir (hafızasız: örnek ömrü üstel dağılır); yine $1/\phi$ olasılıkla rastgele bir **komşunun** rastgele bir örneğini değiştir (uzamsal yayılım: hayaletleri emer, titreşime dayanıklılık).

Artıları: çok hızlı, tamamen paralel sınıflandırma, az son işleme, hayaletleri kendiliğinden siler. Eksileri: dinamik aydınlatma ve kayan arka planlara sabit $R$ ile duyarlılık.
