Boosting'in tek kayba özgü kısmı pseudo-residual. Onu elle türetmek yerine autograd'a bırak:

```python
F = F.detach().requires_grad_(True)
g, = torch.autograd.grad(loss_fn(F, y).sum(), F)
r = -g
```

Başlangıçta verilen `fit_reg_stump` (torch, döngüsüz) ağaç oturtmayı yapıyor; sen yalnız boosting döngüsünü kuruyorsun.

- `pseudo_residuals(F, y, loss_fn)` → $-\partial\sum L/\partial F$.
- `gbm_classify(X, y, rounds=200, lr=0.3)` → model; logistic kayıp (`F.binary_cross_entropy_with_logits`, `reduction="none"`), $F_0=\log\frac{\bar y}{1-\bar y}$.
- `gbm_logits(model, X)` → `(n,)`.

Kontrol: MSE kaybında residual'ın $-2(F-y)$, logistic'te $y-\sigma(F)$ olduğunu ve iç içe çemberleri ≥ 0.9 doğrulukla ayırdığını doğrular (stump'ların toplamı $x_0^2+x_1^2$ gibi toplamsal sınırları temsil edebilir).
