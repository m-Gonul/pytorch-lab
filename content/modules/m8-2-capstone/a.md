EMA: $0.999^{1000}=e^{1000\ln0.999}\approx e^{-1}=0.368$ — "etkin pencere" $1/(1-\beta)=1000$ adım, bu kadar eski ağırlıklar hâlâ ~%37 katkıdadır. 0.995 → 200 adım: kısa eğitimde daha uygun (d aşaması 1200 adım).

CFG'de her adımda koşullu ve koşulsuz iki tahmin: 25 · 2 = 50 değerlendirme; toplu CFG (iki batch'i birleştirip tek çağrı) çağrı sayısını 25'e indirir ama hesap aynıdır (e aşaması).

Zaman embedding'i: $\omega_1=10000^{-0.25}=0.1$, $\sin(5)=-0.959$.

16 → 8 → 4: 4 × 4 = 16 token, skor matrisi 16² = 256 — attention'ı yalnız en düşük çözünürlükte kullanmanın nedeni (16 × 16'da 65 536).

Atlama: yukarı örneklenmiş 2c = 32 kanal + kodlayıcıdan c = 16 → 48.

Dice: 2 · 25 / (30 + 40) = 0.714 (IoU = 25 / 45 = 0.556).
