Üreteç $G(z)$ gürültüden örnek üretir, ayırıcı $D(x)\in(0,1)$ gerçek mi sahte mi tahmin eder:

$$
\min_G\max_D\ V(D,G)=\mathbb E_{x\sim p_\text{data}}[\log D(x)]+\mathbb E_{z}[\log(1-D(G(z)))].
$$

Sabit $G$ için en iyi ayırıcı $D^*(x)=\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)}$ ve bu noktada $V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_\text{data}\,\|\,p_g)$: $G$, Jensen–Shannon ıraksamasını küçültüyor.

Pratikte üreteç $\log(1-D(G(z)))$'yi küçültmek yerine $-\log D(G(z))$'yi küçültür ([[non-saturating loss]]).

GAN eğitimi bir **oyun**: iki oyuncu aynı anda gradyan adımı atar. Bu, tek bir kaybı küçültmekten farklı dinamikler doğurur; en basit örnek **Dirac GAN**: veri $\delta_0$, üreteç $\delta_\theta$, ayırıcı $D(x)=\sigma(\psi x)$. Değer $V(\theta,\psi)=f(\psi\theta)$, $f(t)=-\log(1+e^{-t})$.

WGAN-GP ayırıcıyı ("critic") 1-Lipschitz'e yaklaştırmak için [[gradient penalty]] ekler: $\lambda\,\mathbb E_{\hat x}\big[(\lVert\nabla_{\hat x}D(\hat x)\rVert-1)^2\big]$, $\hat x=\alpha x+(1-\alpha)G(z)$.
