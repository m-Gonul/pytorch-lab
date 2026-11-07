NumPy, **döngü yok**.

**`hog(img, nbins=9, cell=8, block=2)`** — `skimage.feature.hog(img, nbins, (cell, cell), (block, block), block_norm="L2-Hys", feature_vector=True)` ile aynı (b'deki tanım, tüm görüntü; artakalan kenar pikselleri atılır; bloklar 1 hücre adımla, sıra `(blok_y, blok_x, hücre_y, hücre_x, yön)`). Hücre histogramları için ofsetli `np.bincount`; bloklar için `sliding_window_view`.

**`fast_hessian(img, L=9)`** — SURF det yanıtı, $L=9+6k$, lob $l=L/3$:

- $D_{yy}$: üst üste üç $l\times(2l-1)$ şerit, ağırlıklar $+1,-2,+1$ (merkezli, $L\times(2l-1)$); $D_{xx}$ devriği.
- $D_{xy}$: merkezin çevresinde 1 piksellik boşlukla dört $l\times l$ kare; sol üst ve sağ alt $+1$, sağ üst ve sol alt $-1$.
- Kutu toplamları **integral görüntüden** (sıfır dolgulu; görüntü dışı 0); her yanıt $L^2$'ye bölünür; $\det=D_{xx}D_{yy}-(0.9D_{xy})^2$.
