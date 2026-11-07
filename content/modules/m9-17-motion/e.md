Çevrimdışı video stabilizasyonu.

- `estimate_motion(frames)` → `(T-1, 3)` `[dx, dy, da]`: ardışık kareler arası benzerlik dönüşümü: `cv2.goodFeaturesToTrack(prev, 300, 0.01, 10)`, `cv2.calcOpticalFlowPyrLK`, `cv2.estimateAffinePartial2D` (RANSAC); `dx, dy = M[:, 2]`, `da = atan2(M[1, 0], M[0, 0])`. (Kare döngüsü serbest.)
- `smooth(traj, r=5)` — `(T, 3)` yörüngenin `2r+1`'lik hareketli ortalaması, uçlar `edge` dolgulu, **`cumsum` ile döngüsüz**.
- `stabilize(frames, r=5)` → `(out, corr)`: yörünge = `cumsum` (ilk satır 0), düzeltme = yumuşak − yörünge; her kare için $M_t=\begin{bmatrix}\cos a&-\sin a&dx\\\sin a&\cos a&dy\end{bmatrix}$ (düzeltme), **tüm kareler tek** `F.affine_grid` + `grid_sample` çağrısıyla (`M`'yi normalize koordinatlara çevirip tersini al). `out (T, H, W)` `float32`.
