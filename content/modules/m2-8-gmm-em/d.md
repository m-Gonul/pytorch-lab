EM'e alternatif: log-likelihood'u doğrudan gradyanla maksimize et. Kısıtlar yeniden parametrelendirmeyle çözülür: $\pi=\text{softmax}(a)$, $\Sigma_k=L_kL_k^\top$, $L_k$ alt üçgen ve köşegeni $e^{(\cdot)}$ ile pozitif.

```python
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
gmm = MixtureSameFamily(Categorical(logits=a), MultivariateNormal(mus, scale_tril=L))
ll = gmm.log_prob(X).sum()
```

- `make_scale_tril(raw)` — `raw: (K, d, d)` → alt üçgen, köşegeni `exp(diag)`: `torch.tril(raw, -1) + torch.diag_embed(torch.diagonal(raw, dim1=-2, dim2=-1).exp())`.
- `gmm_mean_ll(X, logits, mus, raw)` → ortalama log-likelihood.
- `fit_gmm_gd(X, mus0, steps=800, lr=0.05)` → son ortalama log-likelihood (float); `logits` ve `raw` sıfırdan, `Adam`.

Kontrol, `MixtureSameFamily`'nin c'deki formülünle aynı sayıyı verdiğini ve gradyanla bulunan çözümün EM'inkine yakın olduğunu doğrular.
