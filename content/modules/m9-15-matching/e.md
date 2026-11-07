Panorama, PyTorch'ta toplu RANSAC ile.

- `ransac_t(src, dst, thr=3.0, iters=1000, seed=0)` — c'deki `ransac`'ın torch hali (`torch.Generator().manual_seed(seed)`, `torch.rand(iters, N).argsort(1)[:, :4]`, yığın `torch.linalg.svd`); içerdeğerlerle Hartley normalize DLT. `(H, mask)` (`float64` tensör, bool).
- `stitch(img1, img2)` → `(pano, H)`: `cv2.SIFT_create()`, oran testi 0.75, `ransac_t` ile $H$ (img2 → img1). Tuval: iki görüntünün köşelerini kapsayan dikdörtgen (ötelemeyle negatif koordinatları kaydır); her tuval pikselini `grid_sample` ile her iki görüntüden örnekle (`align_corners=True`, geçerlilik maskesiyle); **tüy (feather) harmanlama**: ağırlık = görüntü kenarına uzaklık (`cv2.distanceTransform` ya da kapalı form), ağırlıklı ortalama. `pano` `uint8`.
