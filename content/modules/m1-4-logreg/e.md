$K$ sınıf için en basit yaklaşım **one-vs-rest**: her sınıf için "bu sınıf mı, değil mi?" diye ayrı bir ikili sınıflandırıcı; tahmin, en yüksek logit'i veren sınıf. $K$ modeli döngüyle eğitmek yerine:

$$
Z = XW + b \in\mathbb{R}^{n\times K},\qquad Y_{ik}=[y_i=k]\ \text{(one-hot)},\qquad
\nabla_W L=\frac1n X^\top\big(\sigma(Z)-Y\big).
$$

- `train_ovr(X, y, K, lr=0.5, steps=300)` → `(W: (d, K), b: (K,))`. Kayıp: her sınıfın ortalama BCE'sinin **toplamı**; başlangıç sıfır; tam-batch gradient descent. Yalnız adım döngüsü.
- `predict_ovr(X, W, b)` → `(n,)` sınıf indeksleri.

Kontrol, sonucu K ayrı ikili modelle karşılaştırır ve hız tablosu üretir.
