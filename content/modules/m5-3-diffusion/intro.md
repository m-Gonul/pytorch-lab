**İleri süreç:** $q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}\,x_{t-1},\beta_t I)$, $\alpha_t=1-\beta_t$. Adımlar Gauss olduğu için kapalı form:

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \bar\alpha_t=\prod_{s\le t}\alpha_s,\ \ \varepsilon\sim\mathcal N(0,I).
$$

**Eğitim:** rastgele $t$ ve $\varepsilon$ seç, $x_t$'yi kur, ağ $\varepsilon_\theta(x_t,t)$ gürültüyü tahmin etsin: $\ \mathcal L=\lVert\varepsilon-\varepsilon_\theta(x_t,t)\rVert^2$.

**DDPM örnekleme** ($t=T,\dots,1$):

$$
x_{t-1}=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta\Big)+\sigma_t z,\qquad \sigma_t^2=\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t .
$$

**DDIM (η = 0):** $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta)/\sqrt{\bar\alpha_t}$, $\ x_{\text{önceki}}=\sqrt{\bar\alpha_{\text{önceki}}}\,\hat x_0+\sqrt{1-\bar\alpha_{\text{önceki}}}\,\varepsilon_\theta$.

**[[flow matching]]:** $z\sim\mathcal N(0,I)$'den veriye doğrusal yol $x_t=(1-t)z+tx_1$; ağ hızı öğrenir: $\ \mathcal L=\lVert v_\theta(x_t,t)-(x_1-z)\rVert^2$. Örnekleme: $\dot x=v_\theta(x,t)$'yi $t=0\to1$ Euler ile çöz.

**Kâhin:** veri $\mathcal N(\mu,s^2)$ ise optimal $\varepsilon$-tahmincisi kapalı formda: $\ x_t\sim\mathcal N(\sqrt{\bar\alpha}\mu,\ \bar\alpha s^2+1-\bar\alpha)$ ve

$$
\mathbb E[\varepsilon\mid x_t]=\frac{\sqrt{1-\bar\alpha}\,(x_t-\sqrt{\bar\alpha}\,\mu)}{\bar\alpha s^2+1-\bar\alpha}.
$$

Bununla örnekleyiciyi, eğitim hatasından bağımsız olarak test edebilirsin.
