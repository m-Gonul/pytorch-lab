Bir kayan nokta sayısı $(-1)^s\cdot1.m\cdot2^{e-\text{bias}}$:

| biçim | üs | mantis | en büyük | 1'in üstündeki adım |
|---|---|---|---|---|
| fp32 | 8 | 23 | $3.4\cdot10^{38}$ | $1.2\cdot10^{-7}$ |
| fp16 | 5 | 10 | 65504 | $9.8\cdot10^{-4}$ |
| bf16 | 8 | 7 | $3.4\cdot10^{38}$ | $7.8\cdot10^{-3}$ |

Düşük hassasiyetin üç sorunu:

1. **Taşma:** fp16'da $300^2=\infty$.
2. **Alt taşma:** fp16'da $10^{-8}=0$; küçük gradyanlar kaybolur → **[[loss scaling]]**: kaybı $S$ ile çarp, gradyanları $S$'ye böl.
3. **Kaybolan güncellemeler:** $w=1$ iken $w+10^{-4}=1$ (fp16). Çözüm: ağırlıkların **fp32 master** kopyası; ya da saf bf16 eğitimde **[[stochastic rounding]]**.

**AMP** (`torch.autocast`): matmul/conv gibi işlemler düşük hassasiyette, kayıp/normalizasyon/indirgeme gibi hassas işlemler fp32'de çalışır. Parametreler ve gradyanları fp32 kalır.
