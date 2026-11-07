Laplace katsayısı $\alpha$ bir hiperparametre. 20 değeri döngüyle denemek yerine tüm modelleri tek tensörde kur:

$$
\log\theta^{(a)}_{cv}=\log(N_{cv}+\alpha_a)-\log\Big(\sum_{v'}N_{cv'}+\alpha_aV\Big)\in\mathbb R^{A\times C\times V},
$$

doğrulama skorları `torch.einsum("nv,acv->anc", Xva, log_theta)` `(A, n, C)`.

- `alpha_sweep(Xtr, ytr, Xva, yva, C, alphas)` → doğrulama doğrulukları `(A,)`. **Döngü yok.**

Kontrol sonucu her α için ayrı eğitimle karşılaştırır ve hız tablosu üretir.
