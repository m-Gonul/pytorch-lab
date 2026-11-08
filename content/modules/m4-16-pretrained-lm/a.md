MLM: 512 · 0.15 = 76.8 seçilir; %80'i (61.4) [MASK], %10'u (7.68) rastgele token, %10'u değişmeden kalır. Kayıp yalnız seçilen 76.8 konumda hesaplanır — token başına sinyal nedensel LM'in ~1/7'si (ELECTRA'nın motivasyonu, x3).

BERT-base: embedding $(30522+512+2)\cdot768+2\cdot768=23\,837\,184$; katman $7\,087\,872$ (ViT ile aynı blok) × 12 = 85 054 464; pooler $768^2+768=590\,592$ → **109 482 240** ≈ 110M.

GPT-2: $12\cdot48\cdot1600^2=1.47\cdot10^9$ (+ embedding 50 257 · 1600 ≈ 80M → 1.5B). GPT-3: $12\cdot96\cdot12288^2=1.74\cdot10^{11}$ ≈ 175B.

Ödül modeli: $-\log\sigma(1.5)=0.201$. KL cezalı: $1-0.1\cdot2=0.8$. DPO: $-\log\sigma(0.1\cdot(2-(-1)))=-\log\sigma(0.3)=0.554$; gradyan ağırlığı $\sigma(-0.3)=0.43$ — örtük ödül sırayı ne kadar yanlış tahmin ediyorsa o kadar büyük.
