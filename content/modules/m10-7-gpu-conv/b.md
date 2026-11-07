Saf Python, okuma sayan simülasyonlar (sıfır dolgu: görüntü dışı okunmaz).

- `conv_tile_loads(T, K)` → `(girdi karosu eleman sayısı, çıktı başına okuma)`.
- `direct_conv(img, k)` → `(çıktı, global okuma)` — her çıktı kendi sınır içi komşularını okur; çıktı float (yuvarlamasız).
- `tiled_conv(img, k, T)` → `(çıktı, global okuma)` — her T×T çıktı karosu için halo'lu girdi karosunu (sınır içi kısmını) bir kez "paylaşımlı" listeye kopyala, çıktıları oradan hesapla.
- `tiled_matmul(A, B, T)` → `(C, global okuma)` — her C karosu için her fazda A ve B karolarını (sınır içi elemanları) bir kez oku.
