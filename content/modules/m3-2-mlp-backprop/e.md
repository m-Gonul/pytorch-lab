16 küçük ağı (aynı mimari, farklı başlatma) aynı veride çalıştırmak: döngüyle 16 ileri geçiş ya da **tek** ileri geçiş.

```python
from torch.func import stack_module_state, functional_call
params, buffers = stack_module_state(models)        # her parametre (E, …)
base = copy.deepcopy(models[0]).to("meta")          # yalnız yapıyı taşıyan şablon
f = lambda p, b, x: functional_call(base, (p, b), (x,))
logits = torch.vmap(f, in_dims=(0, 0, None))(params, buffers, X)   # (E, N, C)
```

- `ensemble_logits(models, X)` → `(E, N, C)`. **Döngü yok.**
- `ensemble_predict(models, X)` → `(N,)`; olasılıkların ortalamasının argmax'ı.

Kontrol, sonucu modeller üzerinde döngüyle karşılaştırır ve hız tablosu üretir.
