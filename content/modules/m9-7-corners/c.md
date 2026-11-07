NumPy, **döngü yok**; `cv2.Sobel`, `cv2.boxFilter` serbest.

`cv2.cornerHarris(img, block, ksize, k)` iç yapısı (`corner.cpp`):

1. Ölçek $s=\dfrac{1}{2^{\,ksize-1}\cdot block\cdot255}$ (uint8 girdi için; `ksize=1` iken $2^0$).
2. `dx = cv2.Sobel(f32, cv2.CV_32F, 1, 0, ksize=ksize, scale=s)`, `dy` benzer (varsayılan kenar kuralı).
3. $a,b,c$ = `cv2.boxFilter(dx*dx | dx*dy | dy*dy, -1, (block, block), normalize=False)`.
4. Harris: $ac-b^2-k(a+c)^2$. MinEigenVal: $a'=a/2$, $c'=c/2$, $(a'+c')-\sqrt{(a'-c')^2+b^2}$.

Yaz:

- `corner_harris_cv(img, block, ksize, k)` ve `corner_min_eig_cv(img, block, ksize)` — `float32`, OpenCV ile aynı.
- `harris_points(img, k=0.04, T_frac=0.01, r=3)` — `corner_harris_cv(img, 3, 3, k)` üzerinde `R > T_frac·max(R)` ve $(2r+1)^2$ yerel en büyük (`scipy.ndimage.maximum_filter`, `mode="constant", cval=-inf`); `(N, 2)` `[x, y]` dizisi, yanıta göre azalan.
