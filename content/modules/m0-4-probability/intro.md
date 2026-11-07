Bir model, verinin hangi süreçten geldiğine dair bir varsayımdır. **Likelihood** bu varsayımın gözlenen veriye ne kadar olasılık verdiğidir; parametreler $\theta$ için

$$
L(\theta)=\prod_{i=1}^n p(x_i\mid\theta),\qquad
\ell(\theta)=\log L(\theta)=\sum_{i=1}^n \log p(x_i\mid\theta).
$$

[[mle|Maximum likelihood estimation]] veriyi en olası kılan $\theta$'yı seçer: $\hat\theta=\arg\max_\theta \ell(\theta)$. Optimizasyon kütüphaneleri minimize ettiği için pratikte [[nll|negatif log-likelihood]] (NLL) minimize edilir; ML'deki kayıpların büyük kısmı budur:

| varsayım | NLL | adı |
|---|---|---|
| $y\sim\mathcal N(\hat y,\sigma^2)$ | $\frac{1}{2\sigma^2}(y-\hat y)^2+c$ | MSE (M1.1) |
| $y\sim\text{Bernoulli}(\sigma(z))$ | $-[y\log p+(1-y)\log(1-p)]$ | BCE (M1.4) |
| $y\sim\text{Kategorik}(\text{softmax}(z))$ | $-\log p_y$ | cross-entropy (M1.5) |

Üç dağılımın MLE'si kapalı formda: Bernoulli için $\hat p=\bar y$, Gauss için $\hat\mu=\bar x$ ve $\hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$, kategorik için $\hat p_k=n_k/n$.
