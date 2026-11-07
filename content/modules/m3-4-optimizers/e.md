Yüzlerce parametre tensörlü bir modelde optimizer adımı bir Python döngüsüdür; her tensör için 6–8 ayrı kernel çağrısı. `torch._foreach_*` işlemleri bir **tensör listesini** tek çağrıda işler:

```python
torch._foreach_mul_(exp_avgs, b1)
torch._foreach_add_(exp_avgs, grads, alpha=1 - b1)
torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - b2)
denom = torch._foreach_sqrt(exp_avg_sqs)
torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
```

- `adamw_foreach_(params, grads, exp_avgs, exp_avg_sqs, step, lr, b1, b2, eps, wd)` — tüm listeyi **yerinde** günceller (AdamW). **Parametreler üzerinde döngü yok.** `step` 1'den başlayan adım numarası.

Kontrol, döngülü referansla karşılaştırır ve 500 küçük tensörde hız tablosu üretir.
