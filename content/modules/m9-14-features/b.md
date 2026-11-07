Saf Python, tek bir 16 × 16 görüntü (4 hücre, 1 blok) için HOG — `skimage.feature.hog(img, 9, (8, 8), (2, 2), block_norm="L2-Hys")` ile aynı.

- `gradients(img)` → `(gx, gy)`: merkezi fark $f(y,x+1)-f(y,x-1)$ ve $f(y+1,x)-f(y-1,x)$, görüntü kenarındaki satır/sütunlarda 0.
- `cell_histogram(gx, gy, y0, x0, cell=8, nbins=9)` — hücredeki her pikselin büyüklüğü $\sqrt{g_x^2+g_y^2}$, açısı $\mathrm{atan2}(g_y,g_x)$ derece mod 180, bölme $\min(\lfloor a/(180/n)\rfloor,n-1)$; toplamı hücre alanına böl.
- `l2hys(v, eps=1e-5)` — $v/\sqrt{\lVert v\rVert^2+\varepsilon^2}$, 0.2'de kırp, yine normalize.
- `hog16(img)` — 4 hücrenin histogramlarını (satır sırası) birleştirip `l2hys`; 36 boyut.
