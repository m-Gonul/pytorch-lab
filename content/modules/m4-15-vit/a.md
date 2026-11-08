<!-- çözüm -->

N = (224/16)² = 14² = 196; [class] ile 197. 384'te 24² = 576 (577 token); konum ızgarası 14 × 14 → 24 × 24. Attention skor matrisi $T^2$: $(577/197)^2=8.58$ — çözünürlük 1.71 kat, attention maliyeti 8.6 kat.

ViT-B/16: yama embedding'i $16\cdot16\cdot3\cdot768+768=590\,592$. Katman: MSA $4(768^2+768)=2\,362\,368$, MLP $768\cdot3072+3072+3072\cdot768+768=4\,722\,432$, iki LN $4\cdot768=3\,072$ → **7 087 872**. Toplam: 590 592 + 768 ([class]) + 197 · 768 (konum) + 12 · 7 087 872 + 1 536 (son LN) + 769 000 (baş) = **86 567 656** ≈ 86M.
