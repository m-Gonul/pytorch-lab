**Bozulma modeli:** $g(x,y)=h*f(x,y)+\eta(x,y)$. Bu modülde yalnız gürültü ($h$ özdeşlik): $g=f+\eta$.

**Gürültü yoğunlukları** ($z$ gri seviye):

| model | yoğunluk | nerede |
|---|---|---|
| Gauss | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(z-\bar z)^2/2\sigma^2}$ | elektronik devre, sensör gürültüsü |
| Rayleigh | $\frac2b(z-a)e^{-(z-a)^2/b}$, $z\ge a$ | menzil görüntüleme |
| Erlang (Gamma) | $\frac{a^bz^{b-1}}{(b-1)!}e^{-az}$ | lazer görüntüleme |
| üstel | $ae^{-az}$ | lazer görüntüleme |
| düzgün | $\frac{1}{b-a}$, $a\le z\le b$ | simülasyon |
| tuz-biber | $P_s$ olasılıkla 255, $P_p$ olasılıkla 0 | hatalı pikseller, iletim |

Gürültü parametreleri görüntünün **düz bir bölgesinden** ortalama ve varyansla kestirilir.

**Ortalama filtreler** ($S_{xy}$ pencere, $mn$ piksel): aritmetik $\frac1{mn}\sum g$, geometrik $(\prod g)^{1/mn}$, harmonik $mn/\sum\frac1g$, kontraharmonik $\sum g^{Q+1}/\sum g^Q$ ($Q>0$ biberi, $Q<0$ tuzu temizler).

**[[median filtre]]:** penceredeki değerlerin medyanı. **Merkez ağırlıklı median:** merkez piksel $w$ kez sayılır; ayrıntıyı daha iyi korur.

**Uyarlamalı ortalama** (yerel gürültü azaltma): $\hat f=g-\frac{\sigma_\eta^2}{\sigma_S^2}(g-m_S)$; $\sigma_\eta^2>\sigma_S^2$ ise oran 1'e kırpılır (çıkış yerel ortalama).

**Uyarlamalı median:**

- *Seviye A:* $z_{min}<z_{med}<z_{max}$ ise B'ye git; değilse pencereyi büyüt, $S_{max}$'a ulaşıldıysa $z_{med}$ ver.
- *Seviye B:* $z_{min}<z_{xy}<z_{max}$ ise $z_{xy}$ (piksel gürültü değil), değilse $z_{med}$.

**[[PSNR]]:** $10\log_{10}\frac{R^2}{\text{MSE}}$, 8 bit için $R=255$.
