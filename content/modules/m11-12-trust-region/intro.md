**Optimizasyon olarak öğrenme:** en küçük kare hata, en küçük çapraz entropi, en büyük olabilirlik, en büyük indirimli ödül. İki aile:

- **Doğru arama:** iyileşme yönü bul (gradyan), adım uzunluğu seç (geri izleme, Armijo).
- **Güven bölgesi:** $f$'yi basit bir $\tilde f$ ile yaklaşıkla, $\tilde f$'ye güvendiğin bölge $R_t$ seç, $\tilde x^*=\arg\min_{x\in R_t}\tilde f(x)$.

$\tilde f$ ikinci dereceden Taylor: $\tilde f(x)=f(c)+\nabla f^\top(x-c)+\tfrac12(x-c)^\top H(x-c)$ — $H$ $d_x\times d_x$: büyük modellerde sorun. Taylor $c$ yakınında doğru → bölge hiperküre $\|x-c\|\le\delta$. $H$ pozitif yarı tanımlıysa dışbükey alt problem (basit, global çözüm); değilse iyileşmeyi garanti eden sezgiseller (Cauchy noktası: gradyan yönünde bölge içinde en iyi nokta; dogleg; Steihaug-CG). Oran $\rho=\dfrac{f(x)-f(x+p)}{\tilde f(x)-\tilde f(x+p)}$ yarıçapı yönetir.

**Politikalarda:** mesafeyi davranışla ölç: $\mathrm{KL}(\pi_\theta\|\pi_{\theta+d})\approx\tfrac12d^\top F d$, $F=\mathbb E[\nabla\log\pi\,\nabla\log\pi^\top]$. KL bölgesinde doğrusal amacın çözümü **doğal gradyan** $F^{-1}g$ (Kakade 2001), adım $\beta=\sqrt{2\delta/(g^\top F^{-1}g)}$.

**TRPO:** vekil amaç $L_\pi(\tilde\pi)=\mathbb E_{s,a\sim\pi}\big[\tfrac{\tilde\pi(a|s)}{\pi(a|s)}A_\pi(s,a)\big]$; teorem $\eta(\tilde\pi)\ge L_\pi(\tilde\pi)-C\,\mathrm{KL}^{\max}$, $C=4\epsilon\gamma/(1-\gamma)^2$ → sınırı her adımda iyileştirmek monoton iyileşme verir. Pratikte: $\max L$ s.t. ortalama KL ≤ δ; $s=F^{-1}g$ eşlenik gradyanla (Fisher-vektör çarpımı = KL'nin Hessian-vektör çarpımı, iki kez geri yayılım), sonra KL ve iyileşme koşullu geri izlemeli çizgi araması.

**PPO:** ikinci derece hesap yok; oran $r_t=\pi_\theta/\pi_{\text{eski}}$, amaç $\mathbb E[\min(r_tA_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$ — kötümser alt sınır; aynı veride birkaç epoch mini-batch SGD. **GAE:** $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$, $A_t=\sum_k(\gamma\lambda)^k\delta_{t+k}$: λ = 0 TD (yanlı, düşük varyans), λ = 1 MC (yansız, yüksek varyans).
