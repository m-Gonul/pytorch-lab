PyTorch'ta mini-batch'leri `torch.utils.data.DataLoader` üretir:

```python
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
for xb, yb in loader:   # bir epoch
    ...
```

- `momentum_path_torch(w0, lr, beta, steps)` — $L(w)=(w-3)^2$'yi `torch.optim.SGD(momentum=beta)` ile minimize et, her adımdan sonraki $w$'leri liste olarak döndür. Elle bulduğun $[1.5, 3.0, …]$ çıkmalı; bu, PyTorch'un momentum formülünün seninkiyle aynı olduğunun kanıtı.
- `train_torch(X, y, lr=0.05, batch_size=32, epochs=50, momentum=0.9)` → eğitilmiş `nn.Linear`.
