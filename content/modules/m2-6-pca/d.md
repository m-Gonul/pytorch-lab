Baş bileşeni bir optimizasyon problemi olarak çöz: Rayleigh bölümü $R(v)=\frac{v^\top Cv}{v^\top v}$ ölçekten bağımsız, maksimumu $\lambda_1$.

- `pca_torch(X, k)` → `(components, variances)`; `torch.linalg.eigh` (artan sırada döner; ters çevir) + işaret kuralı.
- `top_component_gd(X, steps=500, lr=0.05)` → birim `v`; $-R(v)$'yi `Adam` ile minimize et (autograd), sonunda normalize + işaret kuralı.
