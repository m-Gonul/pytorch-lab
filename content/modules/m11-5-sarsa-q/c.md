**Maksimizasyon yanlılığı** (Sutton Örnek 6.7), NumPy, tüm deneyler aynı anda; yalnız bölüm döngüsü serbest.

A durumunda iki eylem: **sağ** → terminal, ödül 0; **sol** → B, ödül 0. B'de `n_b` eylem, her biri terminale gider ve ödül $N(-0.1,1)$. Doğru karar: sağ. γ = 1, ε-açgözlü (ε = 0.1), α = 0.1, Q başlangıçta 0, eşitlikler rastgele.

- `eps_greedy_probs(Q, eps)` — `(n, A)` olasılıklar (açgözlü eylem `argmax`).
- `max_bias(n_runs, n_eps, alpha=0.1, eps=0.1, n_b=10, seed=0, double=False)` — her bölümde sol seçen deney oranı `(n_eps,)`. Çift Q'da davranış $Q_1+Q_2$'ye göre ε-açgözlü; her deney kendi yazı-turasıyla hangi tabloyu güncelleyeceğini seçer.
