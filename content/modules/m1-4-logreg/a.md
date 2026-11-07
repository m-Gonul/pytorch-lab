Hesap makinesi yerine şu değerleri kullan:

| üstel | logaritma |
|---|---|
| $e^{1}\approx 2.7183$ | $\ln 0.5\approx-0.6931$ |
| $e^{-2}\approx0.1353$ | $\ln 0.7311\approx-0.3133$ |
| | $\ln 0.8808\approx-0.1269$ |

Önce üç olasılığı, sonra üç kaybı hesapla. $\ell_2$ en büyük kayıp: model $x=0$ noktasında tam kararsız ($p=0.5$) ve doğru cevap 1. $\ell_3$ en küçük: model emin ve haklı.

Gradyanın işaretlerini yorumla: $\partial L/\partial w<0$, yani $w$'yi **artırmak** kaybı azaltır (eğimi dikleştir, sınıfları daha keskin ayır).
