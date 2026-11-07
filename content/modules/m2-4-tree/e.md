Boosting (M2.5) her turda örneklere farklı ağırlık verir; bölünme skoru **ağırlıklı** sınıf sayımlarıyla hesaplanır. c'deki cumsum hilesi aynen çalışır: one-hot yerine `one_hot * w[:, None]`.

- `best_split_torch(X, y, C, w=None)` → `(j, t, skor)`; `w` örnek ağırlıkları (`None` ise hepsi 1). **Döngü yok.** Kontrol: tamsayı ağırlıkların satırları kopyalamakla aynı sonucu verdiğini ve c ile eşleştiğini doğrular.

Hız tablosu: $n=2000$, $d=20$ için saf Python `best_split` ile karşılaştırma.
