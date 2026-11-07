- `fuse_conv_bn(conv, bn)` → yeni bir `nn.Conv2d` (bias'lı); eval modunda `bn(conv(x))` ile aynı çıktı. Conv'un bias'ı yoksa 0 say. **Döngü yok** (kanal başına işlemler broadcasting ile). Parametreleri `torch.no_grad()` altında kopyala.

Kontrol rastgele running istatistiklerle eşitliği ve hız kazancını ölçer.
