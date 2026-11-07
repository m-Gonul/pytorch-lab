**Kurulum:** politika $\pi$ sabit, $P$ ve $R$ **bilinmiyor**. Yalnız oynayarak deneyim toplayabiliriz. (Bisiklete binmeyi fiziği bilmeden öğrenmek.) Ajan bileşenleri: değer tabanlı, politika tabanlı, aktör–eleştirmen; modelli ya da modelsiz.

**Monte Carlo değerlendirme:** bölümü sonuna kadar oyna, her ziyaret edilen durum için getiriyi hesapla $G_t=r_t+\gamma r_{t+1}+\cdots$, ortala. Artımlı: $V(s)\leftarrow V(s)+\alpha_n(s)\,[G-V(s)]$, $\alpha_n=1/n(s)$ ile tam ortalama. İlk ziyaret / her ziyaret. Yansız, ama getiri birçok rastgele adımın toplamı olduğu için **varyansı yüksek**; çok bölüm ister ve yalnız bölümlü görevlerde çalışır.

**TD(0):** tek adım oyna, kalan getiriyi kendi kestirimimizle tamamla (önyükleme):

$$V(s)\leftarrow V(s)+\alpha\,\big[\underbrace{r+\gamma V(s')-V(s)}_{\delta}\big].$$

Yanlı (V yanlışken hedef de yanlış), **düşük varyans**, her adımda öğrenir. Yakınsama (tablo) Robbins–Monro koşullarıyla: $\sum_n\alpha_n=\infty$, $\sum_n\alpha_n^2<\infty$ (ör. $\alpha_n=1/n$).

**Toplu güncelleme:** sınırlı veriyi tekrar tekrar sunarsak MC ve TD **farklı** cevaplara yakınsar: MC eğitim verisindeki getirilerin ortalamasını, TD ise verinin en çok olabilirlikli Markov modelinin değerini ("kesinlik eşdeğerliği"). Markov ortamlarda TD genelde daha iyi genelleştirir.

**Arası:** n adımlı getiri $G^{(n)}_t=r_t+\cdots+\gamma^{n-1}r_{t+n-1}+\gamma^nV(s_{t+n})$; λ-getirisi $G^\lambda_t=(1-\lambda)\sum_n\lambda^{n-1}G^{(n)}_t$. Geriye bakış: **uygunluk izi** $e(s)\leftarrow\gamma\lambda e(s)+\mathbb 1[s=s_t]$, her adımda $V\leftarrow V+\alpha\delta_te$. λ = 0 → TD(0), λ = 1 → MC.
