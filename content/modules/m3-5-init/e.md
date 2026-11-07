Kritik ölçeği deneyerek bul: ağırlıklar $W=\frac{s}{\sqrt{n}}\,G$, $G$ standart normal. ReLU için teori $s^*=\sqrt2$. 64 farklı $s$'yi döngüyle denemek yerine bir batch boyutu yap: aktivasyonlar `(S, N, width)`, her katmanda `H = act(scales[:, None, None] / √width * (H @ G_l.T))`.

- `std_vs_scale(X, Gs, scales, act)` — `Gs: (depth, width, width)` tüm ölçekler için ortak yön; → `(S, depth)` katman başına std. Yalnız katman döngüsü; ölçekler üzerinde döngü yok.

Kontrol tek tek simülasyonla karşılaştırır ve 30 katmandan sonra std'yi en iyi koruyan ölçeği raporlar.
