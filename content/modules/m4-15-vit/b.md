Saf Python.

- `n_patches(H, W, P)`.
- `vit_params(D, L, M, P, C, img, classes)` — yama embedding'i + [class] + konum + L katman + son LN + doğrusal baş.
- `attn_flops(T, D)` — bir öz-attention katmanının skor ve ağırlıklı toplam kısmı: $2T^2D$ çarpma-toplama (QKᵀ ve AV).
- `patchify(img, P)` — tek kanallı görüntü (liste listesi) → yama listesi, yamalar satır satır (row-major), her yama satır satır düzleştirilmiş.
