One-hot: $\|e_i-e_j\|^2=1+1=2$. OOV: 10 000 + 100. Embedding: 20 × 10 = 200 (bias yok).

Çiftler: konum 0 → 2 bağlam, 1 → 3, 2 → 4, 3 → 3, 4 → 2: **14**.

Kayıp: $-\log\sigma(1)-\log\sigma(0)-\log\sigma(1)=2\cdot0.3133+0.6931=1.3197$ (ikinci negatifin skoru −1 → $-\log\sigma(+1)$).

Gradyan: $\partial J/\partial v_c=(\sigma(u_o^\top v_c)-1)u_o+\sum_k\sigma(u_k^\top v_c)u_k=(0.731-1)\cdot1+0.5\cdot0+0.269\cdot(-1)=-0.538$. $\partial J/\partial u_o=(\sigma-1)v_c$, $\partial J/\partial u_k=\sigma(u_k^\top v_c)v_c$.

Gürültü: $81^{3/4}=27$, $16^{3/4}=8$ → 27/35 = 0.771 (ham sayılarla 81/97 = 0.835: düzleşme).

Alt-örnekleme: $1-\sqrt{10^{-5}/10^{-3}}=1-0.1=0.9$.
