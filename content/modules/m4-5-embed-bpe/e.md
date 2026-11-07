`F.embedding_bag(ids, weight, offsets, mode="mean")`: düz bir id dizisi ve her grubun başlangıç `offsets`'i; her grubun embedding ortalaması. Metin sınıflandırmada ve öneri sistemlerinde "kelime torbası" bu.

- `embedding_bag_mean(weight, ids, offsets)` → `(B, d)`. **Döngü yok:** grup uzunlukları `torch.diff(offsets, append=…)`, her id'nin grup numarası `torch.repeat_interleave(torch.arange(B), lengths)`, toplam `zeros(B, d).index_add_(0, seg, weight[ids])`. Boş grup → sıfır vektör.
