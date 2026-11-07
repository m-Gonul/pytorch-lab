C++'ta üç matmul, aynı derleme biriminde:

- `matmul_ijk(A, B)` — klasik üçlü döngü, en içte $k$: $B$'yi sütun boyunca (stride $n$) okur.
- `matmul_ikj(A, B)` — en içte $j$: hem $B$'nin satırı hem $C$'nin satırı ardışık okunur/yazılır (önbellek dostu, derleyici vektörleştirebilir).
- `matmul_tiled(A, B, T)` — $i$, $k$, $j$ tile'ları + içte ikj.

Kontrol doğruluğu sınar ve `torch.matmul` (BLAS) ile süreleri karşılaştırır.
