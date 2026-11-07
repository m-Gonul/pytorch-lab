Hareketli ortalamanın dört yolu var; hangisinin hızlı olduğunu **ölçmeden** söyleyemezsin:

1. Python döngüsü,
2. `x.unfold(0, w, 1).mean(1)` — $O(nw)$ iş, kopyasız view,
3. `cumsum` farkı: $S_i=\sum_{k\le i}x_k$, ortalama $(S_{i+w}-S_i)/w$ — $O(n)$ iş,
4. `F.conv1d` ile $1/w$'lik çekirdek.

- `moving_average_fast(x, w)` — 2, 3 ya da 4; **döngü yok**. (3'te kayan nokta birikimine dikkat: uzun dizilerde `float64` kullan.)
- `time_fn(fn, *args)` → saniye cinsinden medyan süre; `torch.utils.benchmark.Timer(stmt="fn(*args)", globals={...}).blocked_autorange().median`. Bu sınıf ısınmayı, tekrar sayısını ve (GPU'da) senkronizasyonu yönetir.

Kontrol, `time_fn`'in bağımsız bir ölçümle tutarlı olduğunu doğrular ve dört yöntemi iki farklı $w$ için yarıştırır: $w$ büyüdükçe kazanan değişir.
