$f:\mathbb R^n\to\mathbb R^m$ için tam Jacobian:

- **reverse:** $m$ tane VJP, her biri bir satır: `vjp_fn(e_i)`; birim vektörler üzerinden `vmap`.
- **forward:** $n$ tane JVP, her biri bir sütun: `jvp(f, (x,), (e_j,))`; birim vektörler üzerinden `vmap`.

```python
from torch.func import vjp, jvp
y, vjp_fn = vjp(f, x)
```

- `jacobian_rev(f, x)` → `(m, n)`; `vjp` + `vmap`, **döngü yok**.
- `jacobian_fwd(f, x)` → `(m, n)`; `jvp` + `vmap`, **döngü yok**.

Kontrol, `torch.func.jacrev`/`jacfwd` ile karşılaştırır ve iki uç durumu ($n=8,m=1024$ ve $n=1024,m=8$) yarıştırır.
