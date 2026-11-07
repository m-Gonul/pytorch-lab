Kapıları hesapla ve hücre durumunu güncelle. Forget kapısı $0.73$: eski belleğin $\%73$'ü korunuyor; input kapısı yeni adaydan yarısını ekliyor.

Gradyan için yalnız $c$ yolunu düşün: $\partial c_t/\partial c_{t-1}=f$. $f=0.9$ ile 10 adım sonra gradyan $0.35$; sade RNN'de (her adımda en fazla $0.25$ çarpanlı sigmoid ya da $\lVert W\rVert<1$ ile tanh) $10^{-6}$ mertebesine iner.

GRU'da $z$ eski durumla yeni aday arasında bir interpolasyon: $z=1$ ise hiçbir şey değişmez.
