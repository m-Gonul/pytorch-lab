PyTorch'ta aynı açılım; ek olarak hazır `torch.cdist` var. Sıralamanın tamamı gerekmez, yalnız en küçük $k$: `torch.topk(D, k, largest=False)` $O(n)$ seçim yapar, tam sıralama $O(n\log n)$.

- `sqdist_torch(Xte, Xtr)` — açılım ile, `clamp_min(0)`.
- `knn_predict_torch(Xte, Xtr, ytr, k, C)` — `sqdist_torch` + `topk` + `F.one_hot(...).sum(1).argmax(1)`.

Kontrol, `sqdist_torch`'u `torch.cdist(...) ** 2` ile karşılaştırır ve 3 sınıflı bir veri setinde doğruluğa bakar. Gradyan yok: k-NN'in öğrenilen parametresi yok.
