fp16'da $[1,2)$ aralığında ardışık sayılar $2^{-10}$ arayla dizilir. $10^{-4}$ bunun yarısından küçük, yani en yakına yuvarlama $1$'i geri verir. Bu bir "küçük hata" değil: güncelleme **tamamen** kaybolur ve eğitim durur.

Loss scaling'de pratik değerler $2^{10}$–$2^{16}$; ölçek dinamik olarak ayarlanır.
