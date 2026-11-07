NumPy'da kararlı sigmoid için `if` yerine iki dalı birden vektörel hesaplayıp seçeriz. Ama dikkat: `np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))` **her iki** dalı da tüm elemanlar için hesaplar ve büyük $|z|$'de taşma uyarısı verir. Taşmayan hile: $e^{-|z|}$ her zaman $\le1$.

$$
q=e^{-|z|},\qquad \sigma(z)=\begin{cases}1/(1+q) & z\ge0\\ q/(1+q) & z<0\end{cases}
$$

- `sigmoid_np(z)` — taşma uyarısı **yok** (kontrol, `np.errstate(over="raise", invalid="raise")` altında çağırır; $e^{-1000}$'in 0'a yuvarlanması — underflow — zararsızdır ve serbest).
- `fit_logreg(X, y, lr=0.5, steps=500)` → `(w, b, losses)`; yalnız adım döngüsü.
- `predict(X, w, b)` → 0/1 tamsayı dizisi.
