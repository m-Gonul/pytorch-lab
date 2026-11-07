`loss.backward()` batch kaybının **toplam** gradyanını verir. Bazen her örneğin ayrı gradyanı gerekir: differential privacy (örnek başına kırpma), veri değerlendirme (hangi örnek modeli ne kadar itiyor), meta-öğrenme.

Döngüyle $n$ kez backward ya da `torch.func`:

```python
from torch.func import grad, vmap
def loss_one(w, b, x):          # tek örnek
    return torch.tanh(x @ w + b)
vmap(grad(loss_one), in_dims=(None, None, 0))(w, b, X)   # (n, d)
```

`in_dims` hangi argümanın batch'lendiğini söyler: `w` ve `b` paylaşılıyor (`None`), `X` 0. eksende batch.

- `per_sample_grads(X, w, b)` → `(n, d)`; satır $i$: $\partial\tanh(x_i\cdot w+b)/\partial w$. **Döngü yok.**
