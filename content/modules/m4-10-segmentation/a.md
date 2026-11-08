<!-- çözüm -->

Çıkış: $(7-1)\cdot2-2\cdot1+4=14$.

1B scatter: 1 · [1, 2, 1] konum 0'dan, 2 · [1, 2, 1] konum 2'den, 3 · [1, 2, 1] konum 4'ten → [1, 2, 1 + 2, 4, 2 + 3, 6, 3] = [1, 2, 3, 4, 5, 6, 3]; indeks 4 = **5**. Çekirdek [½, 1, ½] olsaydı tam doğrusal interpolasyon olurdu.

Sıfır ekleme: [1, 0, 2, 0, 3] (uzunluk $(3-1)\cdot2+1=5$), her iki yana $k-1-p=2$ dolgu → 9; 3'lük konvolüsyonla 7 çıkış. ✓

Örtüşme (1B, k = 3, s = 2): çift konumlar 2, tek konumlar 1 katkı alır; 2B'de çarpım: 4, 2, 2, 1 → oran **4**. k = 4, s = 2'de her konum tam 2 (2B'de 4) katkı alır: düzgün.

Bilineer çekirdek (faktör 2): 1B $[0.25, 0.75, 0.75, 0.25]$, dış çarpım merkezi $0.75^2=0.5625$.

U-Net: 572 → 568 → 284 → 280 → 140 → 136 → 68 → 64 → 32 → 28 → (yukarı) 56 → 52 → 104 → 100 → 200 → 196 → 392 → **388**. Kodlayıcıdaki haritalar (568, 280, 136, 64) birleştirmeden önce kırpılır.

IoU $=4/(6+5-4)=4/7$; Dice $=2\cdot4/(6+5)=8/11$. Dice $=2\,\text{IoU}/(1+\text{IoU})$: aynı sıralamayı verir, küçük nesnelerde daha yumuşak.
