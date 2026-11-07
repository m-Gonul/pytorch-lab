- `count_breaks(fn, *args)` — `torch._dynamo.explain(fn)(*args).graph_break_count`.
- `branchy_fixed(x)` — aşağıdaki fonksiyonla aynı sonucu veren, **graph break'siz** sürüm (`torch.where`):

```python
def branchy(x):
    out = x * 2
    if out.sum().item() > 0:
        out = out.tanh()
    else:
        out = out.sigmoid()
    return out + 1
```

- `chain(x)` — 8 ucuz elementwise işlemden (`*`, `+`, `-`) oluşan bir zincir; `compiled_chain = torch.compile(chain)`. Ucuz işlemler bellek-bağımlıdır, füzyonun etkisi en net bunlarda görünür. (`sin`, `exp` ağırlıklı zincirler CPU'da hesap-bağımlı olabilir; orada kazanç küçük, hatta negatif olabilir.)

İlk `torch.compile` çağrısı derleme yapar (CPU'da onlarca saniye sürebilir); kontrol sonra eager ile karşılaştırır.
