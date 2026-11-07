`torch.autograd.Function` ile bir işlemin ileri ve geri geçişini **ayrı ayrı** tanımlarsın:

```python
class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.save_for_backward(y)     # geri geçiş için yalnız bu
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        return grad_out * y          # forward'un her girdisi için bir gradyan

y = Exp.apply(x)
```

Neden?

- **Sayısal kararlılık:** log-sum-exp'in türevi softmax; çıktıyı saklayıp $e^{x-\text{lse}}$ ile hesaplamak taşmaz.
- **Bilerek farklı gradyan:** [[straight-through estimator]] (ileri: round, geri: özdeşlik), gradient reversal (ileri: özdeşlik, geri: $-\lambda g$).
- **Bellek:** autograd'ın sakladıklarını azaltıp geri geçişte yeniden hesaplamak.
- **Birleşik (fused) işlemler:** softmax + cross-entropy'nin gradyanı $(p-y)/N$; ara grafı hiç kurmamak.
