PyTorch optimizer'ları ortak bir taban sınıftan türer:

```python
class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]          # ilk sefer boş sözlük
                ...
```

- `MyAdamW` — `torch.optim.AdamW` ile **aynı** sayıları vermeli (float64, 30 adım, bir MLP'nin tüm parametreleri, iki farklı `param_group`). PyTorch'un formülü: `denom = (v.sqrt() / sqrt(bc2)) + eps`, `p -= lr/bc1 · m / denom`.
