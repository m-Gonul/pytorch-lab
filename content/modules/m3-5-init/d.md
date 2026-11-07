PyTorch'ta başlatma `torch.nn.init` ile, gözlem **hook**'larla:

```python
h = module.register_forward_hook(lambda m, inp, out: ...)   # ileri
out.register_hook(lambda grad: ...)                           # o tensörün gradyanı
h.remove()
```

- `init_model(model, scheme)` — `scheme ∈ {"kaiming", "xavier"}`; tüm `nn.Linear` ağırlıklarına `kaiming_normal_(…, nonlinearity="relu")` ya da `xavier_normal_`, bias'lara 0.
- `layer_stats(model, X)` → `(act_stds, grad_stds)`; her `nn.Linear` çıktısı için std ve o çıktının gradyanının std'si (kayıp: `model(X).pow(2).mean()`), katman sırasıyla. Hook'ları sonunda kaldır.
