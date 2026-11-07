NumPy, döngüsüz (aksi belirtilmedikçe). `boxes (N, 4)`.

- `iou_matrix(A, B)` → `(N, M)`.
- `nms(boxes, scores, thr)` — açgözlü NMS, en fazla bir döngü (IoU'yu dışarıda bir kez hesapla).
- `fast_nms(boxes, scores, thr)` — döngüsüz (YOLACT): skora göre sırala, IoU matrisinin üst üçgenini al, her kutunun kendinden yüksek skorlu herhangi bir kutuyla en yüksek IoU'su eşiği aşıyorsa sil. Bastırılmış kutular da bastırabildiği için tuttuğu küme açgözlü NMS'ninkinin alt kümesidir.
- `ap_all(scores, tp, n_gt)` — sıralama, `cumsum`, `maximum.accumulate` ile zarf.
- `anchor_kmeans(wh, k, iters=30, seed=0)` — YOLOv2: (w, h) çiftlerini uzaklık $1-\text{IoU}$ ile kümele (kutuları aynı merkeze koyarak IoU); başlangıç: rastgele k örnek; tek döngü (iterasyonlar). Çapalar `(k, 2)` ve ortalama en iyi IoU.
