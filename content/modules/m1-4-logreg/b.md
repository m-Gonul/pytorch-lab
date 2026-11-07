Listelerle yaz. `X` satırları örnek olan iç içe liste (`n × d`), `w` uzunluğu `d` olan liste.

- `sigmoid(z)` — **taşmamalı**: $z\ge0$ için $1/(1+e^{-z})$, $z<0$ için $e^{z}/(1+e^{z})$. `sigmoid(-1000)` hata vermeden `0.0` döndürmeli.
- `bce(p, y, eps=1e-12)` — tek örnek kaybı; $p$'yi `[eps, 1-eps]` aralığına kırp.
- `grad(X, y, w, b)` → `(gw, gb, L)`; `gw` liste.
- `fit(X, y, lr=0.5, steps=1000)` → `(w, b)`.

`numpy`/`torch` yasak; `math.exp`, `math.log` serbest.
