Saf Python, kutular `(x1, y1, x2, y2)` demetleri.

- `iou(a, b)`.
- `nms(boxes, scores, thr)` → tutulan indeksler (skor sırasıyla): açgözlü NMS.
- `match(det_boxes, det_scores, gt_boxes, thr=0.5)` → skora göre azalan sırada her tespit için doğru/yanlış listesi; her tespit, henüz eşleşmemiş gerçek kutular arasında IoU'su en yüksek olanla eşleşir (IoU ≥ thr ise doğru; aynı gerçek kutu ikinci kez eşleşemez → yinelenen tespit yanlıştır).
- `pr_curve(tp, n_gt)` → `(precision, recall)` listeleri.
- `ap11(prec, rec)` ve `ap_all(prec, rec)` — 11 noktalı ve tüm noktalı (zarf altındaki alan). Recall düzeylerini `k / 10` ile üret (`0.1 * k` kayan nokta tuzağı: `0.1 * 6 > 0.6`).
