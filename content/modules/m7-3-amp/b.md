Saf Python.

- `round_float(x, man_bits, exp_bits)` — IEEE tarzı biçime en yakına yuvarla (eşitlikte çift: Python `round`). bias $=2^{e-1}-1$; üs $E=\lfloor\log_2|x|\rfloor$ (`math.frexp`), alt sınır $1-\text{bias}$ (subnormal); adım $q=2^{E-m}$; sonuç $\text{round}(|x|/q)\cdot q$; en büyük sonlu değeri aşarsa $\pm\infty$. `0`, `inf`, `nan` olduğu gibi döner.
- `LossScaler(init=2**16, growth=2.0, backoff=0.5, interval=3)` — `update(found_inf)`: inf varsa ölçeği `backoff` ile çarp ve sayacı sıfırla; yoksa sayacı artır, `interval` ardışık iyi adımda ölçeği `growth` ile çarp. `scale` özniteliği.

Kontrol fp16'yı NumPy'ın `float16`'sıyla, bf16'yı bit düzeyinde yuvarlamayla karşılaştırır.
