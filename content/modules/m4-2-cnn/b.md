Saf Python:

- `conv_out(H, k, s, p)` → çıktı boyutu.
- `receptive_field(layers)` — `layers`: `(k, s)` çiftleri listesi.
- `grad_product(local_derivs, residual)` → zincir kuralı çarpımı; `residual=True` ise her katman $1+a$.
- `plan(H, layers)` — `layers`: `(k, s, p)` listesi → her katmandan sonraki boyutlar listesi.
