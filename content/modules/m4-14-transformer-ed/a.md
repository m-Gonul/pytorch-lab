$P_{1,0}=\sin(1/10000^0)=\sin1=0.841$; $P_{2,1}=\cos(2)=-0.416$. Boyut 100 (i = 50): açısal frekans $\omega=10000^{-100/512}$, dalga boyu $2\pi/\omega=2\pi\cdot10000^{0.1953}=37.96$ ≈ 38.

Kodlayıcı katmanı: MHA $4(d^2+d)=1\,050\,624$; FFN $d\,d_{ff}+d_{ff}+d_{ff}\,d+d=2\,099\,712$; iki LN $2\cdot2d=2\,048$ → **3 152 384**. Kod çözücü: iki MHA + FFN + üç LN → **4 204 032**. Toplam: $6\cdot3\,152\,384+6\cdot4\,204\,032+37\,000\cdot512=63\,082\,496$ (makale ≈ 65M; fark sözlük boyu ve sayım ayrıntıları).

Noam tepesi s = w'de: $512^{-0.5}\cdot4000^{-0.5}=1/\sqrt{2\,048\,000}=6.99\cdot10^{-4}$.

Etiket yumuşatma (PyTorch): $q=(1-\varepsilon)\,\text{onehot}+\varepsilon/V$ → doğru sınıf $0.9+0.02=0.92$, diğerleri 0.02. CE $\ge H(q)=-(0.92\ln0.92+4\cdot0.02\ln0.02)=0.390$: model hiçbir zaman olasılık 1 vermeye zorlanmaz — logit'ler sınırlı kalır.

Nedensel maske T = 4: alt üçgen 1 + 2 + 3 + 4 = 10.
