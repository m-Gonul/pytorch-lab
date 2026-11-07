Bir **yumuşak karar düğümü** (soft stump):

$$
g(x)=\sigma\!\Big(\frac{w^\top x+b}{\tau}\Big),\qquad
\text{logit}(x)=g(x)\,\ell_{\text{sağ}}+(1-g(x))\,\ell_{\text{sol}},
$$

$\ell_{\text{sol}},\ell_{\text{sağ}}\in\mathbb R^C$ öğrenilen yaprak logit'leri. $\tau\to0$ iken $g$ bir basamak fonksiyonuna, düğüm sert bir eğik bölünmeye dönüşür.

- `SoftStump(nn.Module)` — `__init__(self, d, C, tau=0.5)`, `forward(x)` → `(n, C)` logit.
- `train_soft_stump(X, y, C, steps=500, lr=0.1)` → eğitilmiş `SoftStump`; `CrossEntropyLoss` + `Adam`.

Kontrol: tek bir özelliğin eşiği sınıfı belirlerken (diğer özellikler gürültü) öğrenilen $w$'nin o özelliğe yoğunlaştığını ve doğruluğun ≥ 0.95 olduğunu doğrular.
