Saf Python; görüntü bir liste listesi (`img[y][x]`), değerler `float`.

- `bilinear_at(img, x, y)` — koordinatlar görüntünün dışına taşarsa kenara kırp (`0 ≤ x ≤ W-1`).
- `resize_nearest(img, H2, W2)` — `cv2.INTER_NEAREST` kuralı: kaynak $\lfloor x\cdot W/W_2\rfloor$.
- `resize_bilinear(img, H2, W2)` — OpenCV'nin piksel merkezi kuralı + `bilinear_at`.

Kontrol, `skimage.data.camera()`'dan kesilen küçük bir parçada `cv2.resize` (float32) ile birebir karşılaştırır.
