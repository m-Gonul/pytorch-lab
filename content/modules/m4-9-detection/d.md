PyTorch, sıfırdan küçük bir YOLO. Görüntüler 64 × 64, 3 sınıf (kare, daire, eşkenar dörtgen), ızgara S = 8 (hücre 8 piksel), 2 çapa.

- `shapes_dataset(n, seed)` → `(X (n,1,64,64), targets)`; hazır (verilir).
- `box_iou(a, b)` — `(N, 4) × (M, 4) → (N, M)`.
- `encode(boxes, labels, S, anchors, C, img=64)` → `(S, S, A, 5 + C)`: merkezin düştüğü hücre; şekil IoU'su en yüksek çapa; `[ofset_x, ofset_y, ln(w/w_a), ln(h/h_a), 1, one-hot]`.
- `decode(p, anchors, img=64, conf=0.5)` — tek görüntü çıkışı `(S, S, A, 5 + C)` → `(kutular, skor = nesnellik × sınıf olasılığı, etiket)`.
- `nms(boxes, scores, thr)`.
- `TinyYOLO(A=2, C=3)` — Conv-BN-LeakyReLU, üç 2× havuz, 1×1 başlık; çıkış `(N, S, S, A, 5 + C)`.
- `yolo_loss(p, t, l_coord=5, l_noobj=0.5)` — sorumlu çapalarda σ(ofset) ve log-ölçek MSE (λ_coord), nesnellik BCE (nesnesiz çapalarda λ_noobj), sınıf CE; batch'e bölünür.
- `mean_ap(dets, gts, C, thr=0.5)` — her görüntü için `(kutular, skorlar, etiketler)`; sınıf başına tüm noktalı AP'lerin ortalaması.
