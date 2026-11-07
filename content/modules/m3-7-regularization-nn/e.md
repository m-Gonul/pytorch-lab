MC dropout'ta $T$ geçişi döngüyle yapmak yerine $T$'yi bir batch boyutu yap (maskeler önceden üretilmiş geliyor, sonuç deterministik):

$$
H_t=\text{ReLU}(XW_1+b_1)\odot\frac{M_t}{1-p},\qquad S_t=H_tW_2+b_2,\qquad t=1..T .
$$

- `mc_logits(W1, b1, W2, b2, X, masks, p)` — `masks: (T, N, H)` bool → `(T, N, C)`. **Döngü yok.**
- `predictive_entropy(logits)` — `(T, N, C)` → `(N,)`; ortalama olasılığın entropisi $-\sum_c\bar p_c\log\bar p_c$.
