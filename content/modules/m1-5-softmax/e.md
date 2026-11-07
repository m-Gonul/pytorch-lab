İki yaygın "döngüyle yazılır ama yazılmamalı" işlem:

1. **Softmax Jacobian'ı.** $J=\partial p/\partial z=\text{diag}(p)-pp^\top$, $(K\times K)$. $n$ örnek için `(n, K, K)` tensörü: `torch.diag_embed(P) - P[:, :, None] * P[:, None, :]`. (Bu, `vmap(jacrev(softmax))` ile aynı şey; kapalı formu bilmek bir kopyayı ve $K$ geri geçişi kurtarır.)
2. **Confusion matrix.** $C_{ij}$: gerçek sınıfı $i$ olup $j$ tahmin edilen örnek sayısı. Döngü yerine her çifti tek bir tamsayıya kodla, $K\cdot i+j$, ve `torch.bincount(…, minlength=K*K)`.

- `softmax_jacobian(Z)` → `(n, K, K)`. **Döngü yok.**
- `confusion_matrix(y_true, y_pred, K)` → `(K, K)` tamsayı. **Döngü yok.**
