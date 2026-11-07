AR(2): 1.2 + 0.3 = 1.5. ARMA(1,1): 1.0 + 0.2 = 1.2. ARIMA: fark [2, 2, 2, 2], tahmin edilen fark 2 → 11 + 2 = 13.

Yule–Walker: $\rho_k=\phi_1\rho_{k-1}+\phi_2\rho_{k-2}$ ($\rho_0=1$, $\rho_{-k}=\rho_k$). $k=1$: $\rho_1=\phi_1+\phi_2\rho_1\Rightarrow\rho_1=\phi_1/(1-\phi_2)=0.6/0.7=0.857$. $k=2$: $\rho_2=0.6\cdot0.857+0.3=0.814$. Tersine, örnek otokorelasyonlardan $\phi$ çözülür: Yule–Walker tahmincisi (c aşaması).

AR(1) varyansı: $\gamma_0=\phi^2\gamma_0+\sigma^2\Rightarrow\gamma_0=\sigma^2/(1-\phi^2)=1/0.36=2.78$; $|\phi|\to1$'de patlar (birim kök: rastgele yürüyüş, durağan değil).

Parametreler (Keras SimpleRNN: $W_x$, $W_h$, tek $b$): $1\cdot16+256+16=288$; $16\cdot16+256+16=528$; $16+1+1=18$ → **834**; yoğun 8: $16\cdot8+8=136$ → **952**. PyTorch `nn.RNN` iki bias tutar ($b_{ih}$, $b_{hh}$): her katmana +h. Model 1: 41, model 2: 3 (Keras).

WaveNet: $1+2\cdot(2^{10}-1)\cdot(k-1)=2047$.
