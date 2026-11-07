NumPy'da **döngüsüz**, batch'li:

1. `np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))`
2. `np.lib.stride_tricks.sliding_window_view(xp, (KH, KW), axis=(2, 3))` → `(N, C, H', W', KH, KW)` kopyasız view; stride için `[:, :, ::S, ::S]`.
3. `np.einsum("nchwkl,fckl->nfhw", windows, w)` + bias.

- `conv2d_np(x, w, b, stride=1, pad=0)` → `(N, F, OH, OW)`.
- `im2col_np(x, KH, KW, stride=1, pad=0)` → `(N, C·KH·KW, OH·OW)`; satır sırası `(c, a, b)`, sütun sırası `(i, j)`. Kontrol, `w.reshape(F, -1) @ im2col` ile conv'un aynı olduğunu doğrular.
