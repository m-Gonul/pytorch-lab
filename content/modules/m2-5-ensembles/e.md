Bagging: $B$ bootstrap örnekleminin her birine bir regresyon stump'ı. Döngü yerine her örneklemi bir **ağırlık vektörü** olarak gör: $w_{b,i}$ = örnek $i$'nin $b$. örneklemde kaç kez çekildiği. Ağırlık matrisi `(B, n)`: `torch.zeros(B, n).scatter_add_(1, idx, ones)`.

Ağırlıklı kazanç $\frac{S_L^2}{W_L}+\frac{S_R^2}{W_R}$; $S$ ağırlıklı residual toplamı, $W$ ağırlık toplamı. Sıralama tüm $b$'ler için aynı (X değişmiyor), cumsum `(B, n-1, d)` tensöründe.

- `bagged_stumps(X, y, B, generator)` → `(j, t, sol, sağ)`, her biri `(B,)`. Bir tarafın ağırlığı 0 olan bölünmeler geçersiz. **Döngü yok.**
- `bagged_predict(model, X)` → `(n,)` ortalama tahmin. **Döngü yok.**
