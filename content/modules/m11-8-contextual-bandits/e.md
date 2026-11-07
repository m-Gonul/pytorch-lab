**Politika dışı değerlendirme**, PyTorch; çok sayıda aday politika aynı anda.

Kayıtlar: `X (n, d)`, eylem `A (n,)`, ödül `R (n,)`, kayıt olasılığı `p (n,)`. Aday politikalar deterministik doğrusal: $\pi_W(x)=\arg\max_kW_k^\top x$, `W (P, K, d)`.

- `policy_actions(W, X)` → `(P, n)`.
- `fit_reward_model(X, A, R, K, lam=1.0)` — eylem başına sırt regresyonu (toplu `linalg.solve`) → `Theta (K, d)`; $\hat r(x,a)=\Theta_a^\top x$.
- `estimates(W, X, A, R, p, Theta)` → sözlük `{"DM", "IPS", "SNIPS", "DR"}`, her biri `(P,)`.

Politika döngüsü yok.
