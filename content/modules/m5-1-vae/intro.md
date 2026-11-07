Gizil değişkenli model: $z\sim p(z)=\mathcal N(0,I)$, $x\sim p_\theta(x\mid z)$. $\log p_\theta(x)$ hesaplanamaz, ama bir kodlayıcı $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}\sigma^2_\phi(x))$ ile alt sınır (ELBO) kurulur:

$$
\log p_\theta(x)\ \ge\ \mathbb E_{q_\phi}\big[\log p_\theta(x\mid z)\big]-\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big).
$$

Gauss için KL kapalı formda: $\ \mathrm{KL}=\tfrac12\sum_j\big(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\big)$.

Beklentiyi tek örnekle tahmin ederken [[reparametrization trick]]: $z=\mu+\sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$. Ağlar genelde $\log\sigma^2$ (`logvar`) üretir; böylece $\sigma>0$ kısıtı kendiliğinden sağlanır.

**[[iwae]]:** $K$ örnekle $\ \log\frac1K\sum_k\frac{p(x,z_k)}{q(z_k\mid x)}$; $K=1$ ELBO'dur, $K\to\infty$ için $\log p(x)$'e yaklaşır.
