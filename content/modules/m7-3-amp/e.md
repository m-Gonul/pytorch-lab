Saf bf16 eğitim (master kopya yok) ve [[stochastic rounding]].

- `sr_bf16(x, generator)` — fp32 tensörü bf16'ya **stokastik** yuvarla, sonucu fp32 olarak döndür. Bit hilesi: `x.view(torch.int32)`'ye $[0,2^{16})$'dan rastgele bir tamsayı ekle, alt 16 biti sıfırla (`& -65536`), `view(torch.float32)`. Döngü yok.
- `train_bf16(X, y, lr, steps, rounding, generator=None)` → son kayıp; ağırlıklar her güncellemeden sonra bf16'ya yuvarlanır: `"nearest"` → `w.to(torch.bfloat16).float()`, `"stochastic"` → `sr_bf16`.

Kontrol: SR yansız mı, sonuçlar bf16'da temsil edilebilir mi, ve SR'li eğitim en yakına yuvarlamalı eğitimden ne kadar iyi.
