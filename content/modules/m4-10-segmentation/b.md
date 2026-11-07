Saf Python.

- `tconv_out_size(H, k, s, p, op=0)`.
- `conv_transpose1d(x, w, s=1, p=0)` — scatter; tam çıkışı üretip baştan ve sondan `p` kırp.
- `conv_transpose2d(x, w, s=1, p=0)` — tek kanal, liste listeleri.
- `overlap_counts(n, k, s)` — `n` girdili 1B transposed conv'da her çıkış konumuna kaç girdinin katkı yaptığı.
- `unet_sizes(H, depth=4)` — valid U-Net: her seviyede iki 3×3 valid konv., 2×2 havuz; kod çözücüde 2× yukarı + iki valid konv. Ara boyutların listesi (girdi dahil, çıkış son eleman).
- `dice_iou(pred, gt)` — ikili maskeler (liste listeleri) → `(dice, iou)`; ikisi de boşsa (1, 1).
