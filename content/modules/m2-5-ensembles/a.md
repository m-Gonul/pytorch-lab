Boosting'i bir tablo hâlinde izle: her satır bir tur, sütunlar $F$, residual ve stump çıktısı. Stump her turda aynı yerden (2.5) bölüyor ve residual'lar her turda yarıya iniyor ($\eta=0.5$). $\eta=1$ olsaydı tek turda sıfır eğitim hatası olurdu; bunun bir aşırı uyum riski olduğunu düşün.

Bagging kısmında korelasyonun etkisine bak: bağımsız 5 model varyansı 5 kat düşürürken $\rho=0.5$ ile yalnız 1.67 kat düşüyor. $B$'yi sonsuza götürsen de $0.5$'in altına inemezsin.
