<!-- çözüm -->

Bağsız: $(784\cdot100+100)+(100\cdot30+30)+(30\cdot100+100)+(100\cdot784+784)=78\,500+3\,030+3\,100+79\,184=163\,814$. Bağlı: kod çözücü ağırlıkları kaybolur, bias'ları kalır: $78\,500+3\,030+100+784=82\,414$ (≈ yarı).

KL: $0.05\ln0.25+0.95\ln(0.95/0.8)=-0.0693+0.1633=0.0939$. Türev: $-\rho/\hat\rho+(1-\rho)/(1-\hat\rho)=-0.25+1.1875=0.9375>0$ → gradyan inişi $\hat\rho$'yi düşürür. Ceza $\hat\rho=\rho$'da 0'dır, $\hat\rho\to0$ ya da 1'de sonsuza gider — L1'den farklı olarak nöronları tamamen öldürmez, hedef orana çeker.

Doğrusal AE: k = 2 kod en büyük iki özdeğerin alt uzayını tutar; hata atılan özdeğerlerin toplamı $1+0.5=1.5$ (Eckart–Young).

Kod çözücü: $(7-1)\cdot2-2\cdot1+3+1=14$.
