Metin sınıflandırması: her belge bir kelime sayım vektörü $x\in\mathbb N^V$. Multinomial NB:

$$
\log p(y=c\mid x)=\log\pi_c+\sum_v x_v\log\theta_{cv}+\text{sabit},\qquad
\theta_{cv}=\frac{N_{cv}+\alpha}{\sum_{v'}N_{cv'}+\alpha V}.
$$

$N_{cv}$, $c$ sınıfındaki belgelerde $v$ kelimesinin toplam sayısı: `F.one_hot(y, C).T.float() @ X` `(C, V)`.

- `fit_mnb(X, y, C, alpha=1.0)` → `(log_prior: (C,), log_theta: (C, V))`.
- `predict_mnb(model, X)` → `(n, C)` normalize log-olasılık; `torch.logsumexp` ile normalize et.

Kontrol, küçük sentetik bir "spam" veri setinde doğruluğa ve elle yazılmış referansa bakar.
