`unfold`'un bir de tensör metodu var ve **kopyasız** view üretir: `x.unfold(2, KH, S).unfold(3, KW, S)` → `(N, C, OH, OW, KH, KW)`. Bunu einsum'a vermek, `F.unfold`'un kopyalayan im2col'una göre bellek tasarrufu sağlar.

- `conv2d_strided(x, w, b, stride=1, pad=0)` — `F.pad` + iki `.unfold` + `torch.einsum`. **Döngü yok.**

Kontrol doğruluğu `F.conv2d` ile karşılaştırır, sonra dört yolu yarıştırır: Python döngüleri (küçük girdide), `F.unfold` + matmul, strided einsum ve `F.conv2d`.
