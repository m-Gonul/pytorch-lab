NumPy'da $C=AB$ tek operatördür: `A @ B` (ya da `np.matmul`). Bu aşamada **döngü ve comprehension yasak**; kontrol bunu kaynak kodun AST'sinden (soyut sözdizimi ağacı) denetler.

İki fonksiyon yaz:

- `matmul_np(A, B)` — `np.ndarray` döndürsün.
- `center_columns(X)` — her sütundan o sütunun ortalamasını çıkarsın. Broadcasting ile tek satır: `X.mean(axis=0)` shape'i `(k,)` ve `(n, k)` ile sağdan hizalanır.

pandas'ın burada yeri küçük: aynı işlem `df - df.mean()`; pandas sütunları **isimleriyle** hizalar, NumPy **pozisyonla**. Kodun altına bunu deneyen bir satır eklemekte serbestsin.
