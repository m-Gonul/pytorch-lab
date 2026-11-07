Gerçek kullanımda bir sorgu kümesi `Q` (`n × d`) ile bir veri tabanı `K` (`m × d`) arasındaki tüm benzerlikler, ya da `n` vektör çiftinin satır satır iç çarpımları gerekir. İkisi de tek `einsum`:

```python
torch.einsum("nd,nd->n", U, V)   # satır satır iç çarpım: (U * V).sum(-1)
torch.einsum("nd,md->nm", Q, K)  # tüm çiftler: Q @ K.T
```

- `rowwise_dot(U, V)` — `(n, d), (n, d) → (n,)`.
- `pairwise_cosine(Q, K)` — `(n, d), (m, d) → (n, m)`.

İkisinde de **döngü yok**. Kontrol, 4096 çift için `for` + `torch.dot` ile vektörize hâlini yarıştırır.
