Yüzlerce küçük veri setinin (ör. her görüntü yaması, her zaman penceresi) PCA'sı: döngü yerine **batched SVD**. `torch.linalg.svd` baştaki boyutları batch sayar.

- `batched_pca(Xs, k)` — `Xs: (B, n, d)` → `(components: (B, k, d), variances: (B, k))`. Ortalama `Xs.mean(1, keepdim=True)`. İşaret kuralı vektörel: `idx = Vk.abs().argmax(-1, keepdim=True)`, `sign = Vk.gather(-1, idx).sign()`. **Döngü yok.**
