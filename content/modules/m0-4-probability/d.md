`torch.distributions` her dağılım için `log_prob` verir; `log_prob` [[autograd]] ile türevlenebilir. Böylece kapalı formu olmayan modellerde de MLE'yi gradyanla yapabiliriz. Burada kapalı form var; amacımız iki yolun aynı yere vardığını görmek.

$\sigma>0$ kısıtını gradient descent'e zorla uydurmak yerine **yeniden parametrelendir**: $\sigma=e^{s}$, $s\in\mathbb R$ serbest.

- `normal_nll(x, mu, sigma)` — `torch.distributions.Normal` ile **ortalama** NLL.
- `fit_normal(x, steps=2000, lr=0.05)` → `(mu, sigma)`; `mu` ve `s` parametrelerini `torch.optim.Adam` ile eğit. Kontrol sonucu $\hat\mu=\bar x$ ve $\hat\sigma=\sqrt{\hat\sigma^2_{\text{MLE}}}$ ile karşılaştırır.
