`loss.backward()` yazdığında PyTorch ne yapıyor? Bu modülde cevabı kendin yazıyorsun. Fikir Karpathy'nin micrograd'ından: her sayı bir `Value` nesnesine sarılır ve üç şeyi hatırlar:

1. `data` — değeri,
2. `_prev` — kendisini üreten girdileri (grafiğin kenarları),
3. `_backward` — kendi yerel türevini girdilerine nasıl dağıtacağını bilen küçük bir fonksiyon.

`L.backward()` grafiği [[topolojik sıralama|topolojik olarak sıralar]], `L.grad = 1` yapar ve düğümleri **ters** sırada gezip her birinin `_backward`'ını çağırır. M0.3'te elle yaptığın her şey, ama otomatik.

Kullanacağımız yerel türevler:

| işlem | ileri | geri (gelen gradyan $g$ ile) |
|---|---|---|
| $o=a+b$ | | $a \mathrel{+}= g$, $b \mathrel{+}= g$ |
| $o=a\cdot b$ | | $a \mathrel{+}= g\cdot b$, $b \mathrel{+}= g\cdot a$ |
| $o=a^k$ | | $a \mathrel{+}= g\cdot k\,a^{k-1}$ |
| $o=\tanh a$ | | $a \mathrel{+}= g\,(1-o^2)$ |
