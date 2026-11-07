MOSSE izleyici (PyTorch, `torch.fft`). Pencere `w × h` (varsayılan 64 × 64), kırpma `cv2.getRectSubPix(frame, (w, h), (cx, cy))`.

- `preprocess(p)` — `log(p + 1)`, ortalama 0 / std 1 (`+1e-5`), `np.outer(np.hanning(h), np.hanning(w))` ile çarp; `float64` tensör.
- `target(h, w, s=2.0)` — merkezi `(h//2, w//2)` olan Gauss yanıtının `fft2`'si.
- `class Mosse(frame, cx, cy, w=64, h=64, lam=1e-2, n_aug=8, lr=0.125, seed=0)` — başlangıç: ilk kırpma + `n_aug − 1` rastgele küçük afin bozulma (`rng = np.random.default_rng(seed)`, açı ∈ [−10°, 10°], ölçek ∈ [0.95, 1.05], `cv2.getRotationMatrix2D((w/2, h/2), …)`, `cv2.warpAffine(…, borderMode=cv2.BORDER_REFLECT)`); $A=\sum GF^*$, $B=\sum FF^*$.
- `update(frame)` → `(cx, cy, psr)`: yanıt `ifft2(F · A/(B+λ)).real`, tepenin merkeze göre ofsetiyle konumu güncelle; PSR: tepe çevresindeki 11 × 11 dışındaki yan lob; yeni konumda kırpıp $A,B$'yi `lr` ile güncelle.

Kontrol, dönen ve hareket eden bir hedef videosunda MOSSE'yi sabit şablonlu NCC ile karşılaştırır.
