PyTorch'ta aynı model üç parça: `nn.Linear` (model), `nn.MSELoss` (kayıp), `torch.optim.SGD` (güncelleme kuralı). Eğitim döngüsü her zaman aynı beş satır:

```python
pred = model(X)              # ileri
loss = loss_fn(pred, y)      # kayıp
opt.zero_grad()              # eski gradyanları sil (yoksa birikir!)
loss.backward()              # geri
opt.step()                   # θ ← θ − η∇L
```

- `one_step_grads()` — elle örnekteki ($x=[1,2,3]$, $y=[2,4,5]$, $w=b=0$) $\partial L/\partial w$ ve $\partial L/\partial b$'yi autograd ile hesaplayıp `(gw, gb)` döndür. Elle bulduğun $-16.67$ ve $-7.33$ çıkmalı.
- `fit_torch(X, y, lr=0.1, steps=500)` — eğitilmiş `nn.Linear` döndür. `y` shape'ine dikkat: `(n,)` ile `(n, 1)` arasında [[broadcasting]] sessizce $(n, n)$ üretir ve kayıp yanlış olur.
