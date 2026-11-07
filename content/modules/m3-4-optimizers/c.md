NumPy'da sözlük tabanlı bir Adam: parametreler `{"W": array, …}`.

- `class Adam` — `__init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False)`; `step(self, grads)` parametreleri **yerinde** günceller. Parametre sözlüğü üzerinde tek döngü serbest; eleman döngüsü yok.

Kontrol: (1) referansla birebir yörünge, (2) aynı $\lambda$ ile Adam (L2) ve AdamW'ın farklı ağırlık normlarına gittiği bir problem, (3) kötü koşullu bir kuadratikte Adam'ın aynı adım sayısında SGD'den çok daha düşük kayba indiği.
