Varyans çarpanını hesapla: standart normal ağırlıklarla 100 girdili bir nöron varyansı 100 kat büyütüyor; 5 katmanda $10^{10}$. Xavier ve Kaiming bu çarpanı 1'e çekiyor.

Sonra ReLU'nun etkisini gör: Xavier + ReLU her katmanda ikinci momenti yarıya indiriyor; 10 katmanda bin kat küçülme. Kaiming'in 2 çarpanı tam bunu düzeltiyor.
