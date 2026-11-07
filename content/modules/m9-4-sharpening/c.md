NumPy (bulanıklaştırma için `cv2.GaussianBlur` serbest, diğer filtreler NumPy ile), **döngü yok**.

- `sharpen_laplacian(img, kind=4)` — `kind` 4 ya da 8 (köşegenli); $g=f-\nabla^2f$, `uint8`'e kırp. Kenar `REFLECT_101`.
- `unsharp(img, k=1.0, ksize=5, sigma=0)` — `cv2.GaussianBlur` ile $\bar f$, $g=f+k(f-\bar f)$, `uint8`'e kırp (`np.rint`).
- `gradient_magnitude(img, norm="l2")` — Sobel ile $g_x,g_y$, `"l2"` ya da `"l1"`; float32.
