PyTorch. Veri: sklearn rakamları 16 × 16'ya büyütülmüş (P = 4 → 16 yama).

- `PatchEmbed(P, C, D)` — `nn.Conv2d(C, D, P, P)`; çıktı `(B, N, D)`. `forward_unfold(x)`: aynı ağırlıklarla `F.unfold` + matris çarpımı (eşitlik testi).
- `ViT(img=16, P=4, D=64, L=4, h=4, classes=10, pool="cls", pos=True)` — `nn.TransformerEncoderLayer(norm_first=True, activation="gelu", batch_first=True)`, MLP oranı 2, dropout 0.1; `pool="gap"` yama token'larının ortalaması; `pos=False` konum embedding'i yok.
- `train_eval(model, Xtr, ytr, Xte, yte, epochs)` — AdamW (1e-3, wd 0.05), batch 32.
