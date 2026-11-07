İkili örnekte üç noktanın margin'ini hesapla. $x=(2,1)$ tam margin sınırında (kayıp 0, gradyan 0): bu bir **destek vektörü**. $(1,1)$ karar sınırının üzerinde ($f=0$): kayıp 1 ve gradyan $w$'yi bu noktayı pozitif tarafa itecek yönde.

Çok sınıflı örnek CS231n'den: kedi resminin skorları [kedi 3.2, araba 5.1, kurbağa −1.7]. Araba, kediyi $\Delta=1$'lik güvenlik payıyla geçmediği için ceza alıyor; kurbağa zaten çok geride. Doğru sınıfın skoru, ihlal eden her sınıf için bir kez aşağı itilir (gradyan $-1$ katı).
