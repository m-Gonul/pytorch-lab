Türevlenebilir histogram eşitleme.

- `soft_histogram(x, bins=64, sigma=None)` — `x` [0, 1] aralığında; her pikselin her kutuya katkısı $\exp(-(x-c_k)^2/2\sigma^2)$ (varsayılan $\sigma$ = kutu genişliği), normalize. Gradyan akar.
- `ToneCurve(n=64)` — monoton artan bir eğri: `softmax(logits)`'in birikimli toplamı, düğümler arası doğrusal interpolasyon; `forward(x)` [0,1] → [0,1].
- `learn_equalization(x, steps=300, lr=0.05)` → `ToneCurve`: çıkışın yumuşak histogramının **entropisini** büyüt (düzgün dağılım = en yüksek entropi).

Kontrol öğrenilen eğrinin monoton olduğunu ve çıkışın **16 kutulu** histogram entropisinin klasik HE'ninkine ulaştığını sınar. (256 seviyede entropi deterministik bir eşlemeyle artamaz: eşitleme seviyeleri birleştirir, ayıramaz. Düzlük kaba kutularda ölçülür.)
