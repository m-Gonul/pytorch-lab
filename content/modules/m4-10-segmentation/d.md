PyTorch. Veri: 64 × 64 tuvalde 2–5 şekil (kare, daire, eşkenar dörtgen; üst üste binebilir), 4 sınıf (0 = arka plan).

- `seg_dataset(n, seed)` — verilir.
- `conv_input_grad(y, w, s, p, H)` — `x ↦ conv2d(x, w)`'nin girdisine göre vektör-Jacobian çarpımı `vjp(y)` (autograd ile); `F.conv_transpose2d(y, w, stride=s, padding=p)` ile aynı olmalı.
- `bilinear_init_(tconv)` — `nn.ConvTranspose2d`'yi (C → C) kanal başına bilineer çekirdekle başlat.
- `FCN(C=4, w=12, skips=False, bilinear=True)` — paylaşılan kodlayıcı (`Encoder`, verilir; 3 havuz, adım 8); `skips=False`: 1×1 skor → ×8 transposed conv (k = 16, p = 4) ("FCN-8 adımlı-32s"); `skips=True`: ×2, + 1×1 skor(f3), ×2, + skor(f2), ×2 (FCN-8s düzeni).
- `UNet(C=4, w=12)` — her seviyede `ConvTranspose2d(k=2, s=2)` + birleştirme + iki Conv-BN-ReLU.
- `miou(pred, gt, C)` ve `edge_acc(pred, gt, r=2)` — etiket sınırına `r` piksel yakındaki piksellerde doğruluk (max/min havuzla).
