IoU: kesişim $[3,5]\times[2,5]$ = 2·3 = 6; birleşim 16 + 16 − 6 = 26 → 0.2308.

NMS: 0.9'luk kutu kalır; (1,1,11,11) ile IoU = 81/119 = 0.68 → silinir. 0.7'lik kalır; (8,8,18,18) ilk kutuyla 4/196 → kalır; (21,19,31,29) ile 0.7'lik arasında 81/119 → silinir. **3 kutu.**

YOLO: 169 · (25 + 20) = 7 605. Merkez: (7 + σ(0)) · 32 = 7.5 · 32 = 240. Boyut: 150 × 45 = 6 750.

AP: precision / recall: 1/0.2, 1/0.4, 0.667/0.4, 0.75/0.6, 0.6/0.6, 0.5/0.6, 0.571/0.8, 0.5/0.8. Zarf: recall ≤ 0.4 → 1, ≤ 0.6 → 0.75, ≤ 0.8 → 4/7, 0.9 ve 1.0 → 0. 11 nokta: (5 · 1 + 2 · 0.75 + 2 · 4/7) / 11 = 0.6948. Tüm noktalar: 0.2 · (1 + 1 + 0.75 + 4/7) = 0.6643 (recall 1'e hiç ulaşılmadığı için son 0.2'lik dilim sıfır katkı).

FCN: 14 − 7 + 1 = 8. FPN: $\sqrt{5000}=70.7$, $\log_2(70.7/224)=-1.66$, $\lfloor 2.34\rfloor=2$.
