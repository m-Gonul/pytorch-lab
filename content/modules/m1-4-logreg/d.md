Kaybı olasılıktan değil doğrudan logit'ten hesaplamak hem daha kararlı hem daha hızlı. $\ln\sigma(z)=-\ln(1+e^{-z})$ ve $\ln(1-\sigma(z))=-z-\ln(1+e^{-z})$ yerine konunca:

$$
\ell(z,y)=\operatorname{softplus}(z)-zy,\qquad \operatorname{softplus}(z)=\ln(1+e^{z}).
$$

$\operatorname{softplus}$'ı taşmadan hesaplamanın iki yolu var ve **ikisi aynı sayıyı verir**:

```python
torch.logaddexp(torch.zeros_like(z), z)            # ln(e⁰ + eᶻ)
z.clamp(min=0) + torch.log1p(torch.exp(-z.abs()))  # max(z,0) + ln(1+e^{-|z|})
```

Ama ikincisinin **gradyanı** $z=0$'da yanlış: `clamp` ve `abs` iki ayrı kırılma noktası; autograd her birinde bir alt-gradyan seçer ve seçimler birbirini tutmaz. Gerçek fonksiyon pürüzsüz ve türevi $\sigma(0)=0.5$, autograd ise 0 ya da 1 bulur. Kontrol $z=0$'ı test eder.

- `bce_logits(z, y)` — ortalama kayıp. Hazır BCE fonksiyonlarını kullanma; kontrol seninkini `F.binary_cross_entropy_with_logits` ile karşılaştıracak, $z=\pm100$ ve $z=0$ dahil.
- `train_torch(X, y, lr=0.5, steps=500)` → eğitilmiş `nn.Linear(d, 1)`; `nn.BCEWithLogitsLoss` ve `optim.SGD` ile.

Kontrol ayrıca autograd'ın `bce_logits` üzerinden bulduğu $\partial L/\partial z$'nin $(\sigma(z)-y)/n$ olduğunu doğrular.
