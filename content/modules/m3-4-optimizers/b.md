Saf Python; parametre ve gradyan **liste**, durum bir **sözlük** (ilk çağrıda boş gelir, fonksiyon kendisi doldurur). Her fonksiyon yeni parametre listesini döndürür.

- `sgd_momentum(w, g, state, lr, beta=0.9)`
- `rmsprop(w, g, state, lr=0.01, alpha=0.99, eps=1e-8)`
- `adam(w, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False)` — `decoupled=False`: `g += wd·w` (Adam); `True`: `w *= 1 - lr·wd` (AdamW). Adım sayacı `state["t"]`.

Kontrol, yörüngeleri PyTorch formüllerinden üretilmiş referansla karşılaştırır.
