`MemEffSiLU` — $\operatorname{silu}(x)=x\,\sigma(x)$. Türev: $\sigma(x)\big(1+x(1-\sigma(x))\big)$.

`forward`'da **yalnız `x`'i** sakla; $\sigma(x)$'i geri geçişte yeniden hesapla. Naif `x * torch.sigmoid(x)` üç tensör saklar (sigmoid çıktısı ve mul'un iki girdisi).

`mem_eff_silu(x)` = `MemEffSiLU.apply(x)`. Kontrol gradcheck'i, saklanan baytı ve süreyi karşılaştırır.
