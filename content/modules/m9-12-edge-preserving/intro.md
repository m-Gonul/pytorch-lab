**AHE:** tek bir küresel dönüşüm yerine her pikseli **kendi komşuluğunun** histogramıyla eşitle. Çıkış, pikselin komşuluğundaki sırasıyla orantılıdır. Sorun: düz bir bölgede histogram dar bir tepedir, CDF dik çıkar ve küçük gürültü büyük kontrasta dönüşür. Kenar pikselleri için komşuluk yansıtmayla genişletilir (sıfır dolgu ya da kopyalama histogramı bozar).

**CL (kontrast sınırlama):** dönüşümün eğimi histogram yüksekliğiyle orantılıdır, o halde histogramı bir **kırpma eşiğinde** kes ve fazlayı bölmelere eşit dağıt (bazı bölmeler yeniden eşiği aşabilir; istenirse işlem tekrarlanır).

**Döşeme + enterpolasyon:** görüntüyü $8\times8$ (16, 32) döşemeye böl, her döşemenin LUT'unu bir kez hesapla. Her piksel en yakın dört döşeme merkezinin LUT'larıyla dönüştürülüp bilineer karıştırılır; kenar şeridinde iki, köşelerde bir LUT. Kayan pencere (SWAHE) daha doğru ama çok daha pahalıdır; döşemeler bağımsız olduğu için CLAHE paralelleşir.

**OpenCV CLAHE ayrıntıları:** görüntü döşeme sayısına bölünmüyorsa sağ ve alttan `BORDER_REFLECT_101` ile **(tiles − boyut mod tiles)** kadar doldurulur (bölünen eksen de!). Kırpma eşiği $\max(\lfloor\text{clip}\cdot N_{tile}/256\rfloor,1)$; fazlalık önce $\lfloor e/256\rfloor$ olarak her bölmeye, kalan $r$ adet $\max(\lfloor256/r\rfloor,1)$ adımla baştan dağıtılır. LUT $=\mathrm{round}(\mathrm{cdf}\cdot255/N_{tile})$. Enterpolasyon `float32`'de: $t_x=x/w_t-0.5$.

**Bilateral:** $BF[I]_p=\frac1{W_p}\sum_qG_{\sigma_s}(\lVert p-q\rVert)\,G_{\sigma_r}(|I_p-I_q|)\,I_q$. Uzaysal çekirdek yalnız mesafeye, değer çekirdeği yalnız yoğunluk farkına bakar; kenarın öbür yanındaki pikseller değer çekirdeğinde sönümlenir. $\sigma_s$: komşuluk ölçeği (ör. köşegenin %2'si), $\sigma_r$: "kenar sayılan en küçük genlik". Kalın özellikler korunur, $\sim2\sigma_s$'den ince özellikler ortalanır.

**NLM:** her piksel için arama penceresindeki tüm $q$'lar; benzerlik tek değer yerine **yamalar** arasında: $d^2(p,q)=\frac1{|P|}\lVert V_p-V_q\rVert^2$, $w=e^{-\max(d^2-2\sigma^2,0)/h^2}$, $\hat f(p)=\sum_qw\,g(q)/\sum_qw$. Uzaklık cezası yok.

**Yöntem gürültüsü:** temiz görüntüye süzgeci uygula, farkı ($f-D(f)$) incele. Mükemmel süzgeç temiz görüntüye dokunmaz; fark ne kadar küçük ve yapısızsa o kadar iyi.
