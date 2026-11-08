Üreteç $G(z)$ gürültüden örnek üretir, ayırıcı $D(x)\in(0,1)$ gerçek mi sahte mi tahmin eder:

$$
\min_G\max_D\ V(D,G)=\mathbb E_{x\sim p_\text{data}}[\log D(x)]+\mathbb E_{z}[\log(1-D(G(z)))].
$$

Sabit $G$ için en iyi ayırıcı $D^*(x)=\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)}$ ve bu noktada $V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_\text{data}\,\|\,p_g)$: $G$, Jensen–Shannon ıraksamasını küçültüyor.

Pratikte üreteç $\log(1-D(G(z)))$'yi küçültmek yerine $-\log D(G(z))$'yi küçültür ([[non-saturating loss]]).

GAN eğitimi bir **oyun**: iki oyuncu aynı anda gradyan adımı atar. Bu, tek bir kaybı küçültmekten farklı dinamikler doğurur; en basit örnek **Dirac GAN**: veri $\delta_0$, üreteç $\delta_\theta$, ayırıcı $D(x)=\sigma(\psi x)$. Değer $V(\theta,\psi)=f(\psi\theta)$, $f(t)=-\log(1+e^{-t})$.

WGAN-GP ayırıcıyı ("critic") 1-Lipschitz'e yaklaştırmak için [[gradient penalty]] ekler: $\lambda\,\mathbb E_{\hat x}\big[(\lVert\nabla_{\hat x}D(\hat x)\rVert-1)^2\big]$, $\hat x=\alpha x+(1-\alpha)G(z)$.

:::türetim En iyi ayırıcı ve Jensen–Shannon bağlantısı (adım adım)
Değer fonksiyonu $V(D,G)=\mathbb E_{p_d}[\log D(x)]+\mathbb E_{p_g}[\log(1-D(x))]=\int\big[p_d(x)\log D(x)+p_g(x)\log(1-D(x))\big]dx$.

**1. Sabit G için en iyi D:** integrali her $x$ için ayrı en büyükleyebiliriz (D her noktada serbest seçilebilen bir fonksiyon). $a\log y+b\log(1-y)$'nin $y\in(0,1)$'de türevi $\frac ay-\frac b{1-y}=0\Rightarrow y=\frac a{a+b}$. Dolayısıyla $D^*(x)=\frac{p_d(x)}{p_d(x)+p_g(x)}$.

**2. Yerine koy:** $V(D^*,G)=\mathbb E_{p_d}\big[\log\frac{p_d}{p_d+p_g}\big]+\mathbb E_{p_g}\big[\log\frac{p_g}{p_d+p_g}\big]$. Karışım $m=\frac{p_d+p_g}2$ ile paydaları $2m$ yaz: $V=-\log4+\mathrm{KL}(p_d\|m)+\mathrm{KL}(p_g\|m)=-\log4+2\,\mathrm{JS}(p_d\|p_g)$.

**3. Sonuç:** JS ≥ 0 ve yalnız $p_g=p_d$'de 0 olduğundan üreticinin küresel optimumu $p_g=p_d$, değer $-\log4$, ve orada $D^*\equiv\frac12$.

**Varsayımlar ve neden pratikte bozulur:** (i) D sonsuz kapasiteli ve her adımda optimuma eğitiliyor varsayılır; gerçekte ikisi de sınırlı ağ ve birkaç gradyan adımı. (ii) Destekler örtüşmezse (düşük boyutlu manifoldlar) JS sabit $\log2$'dir ve $G$'ye gradyan vermez — orijinal kayıpta kaybolan gradyan; "doymayan" üretici kaybı $-\log D(G(z))$ ve Wasserstein GAN buna yanıttır. (iii) Minimax'ın sırası (önce max) eşzamanlı gradyan adımlarında korunmaz; dinamik dönebilir (mod çöküşü, M5.7).
:::
