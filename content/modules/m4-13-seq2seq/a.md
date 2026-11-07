Beam: 0.75 · 0.32 = 0.24 ("How will" 0.27, "How do" 0.12). Her adımda 3 × 10 000 = 30 000 aday puanlanır, en iyi 3 tutulur.

Ağaç: açgözlü önce A'yı (0.6) seçer, sonra x'i (0.4): 0.24. k = 2: ilk adımda {A 0.6, B 0.4}; ikinci adımda adaylar Ax 0.24, Ay 0.18, Az 0.18, Bx 0.36 → en iyi **Bx = 0.36**. İlk adımda düşük olasılıklı dal, sonradan kazanır.

Uzunluk: $\log 0.2/4=-0.402$; 2 token'lık 0.3'lük hipotez: $\log0.3/2=-0.602$. Normalizasyonsuz ($\log p$) kısa hipotez kazanır (−1.20 > −1.61): olasılıkların çarpımı her token'da küçüldüğü için beam search kısa çıktıları kayırır.

Luong: skorlar $s\cdot y_i=1,0,1$ → $\alpha=(e,1,e)/(2e+1)=(0.422, 0.155, 0.422)$; bağlam $=\alpha_1[1,0]+\alpha_2[0,1]+\alpha_3[1,1]=[2e,e+1]/(2e+1)$ → ilk bileşen 0.845.
