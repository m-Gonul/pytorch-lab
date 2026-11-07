`bf16_saved(fn, *args)` — `fn(*args)`'ı, saklanan tüm kayan noktalı tensörleri bf16'ya **paketleyerek** çalıştır, çıktıyı döndür. `pack`: `(t.dtype, t.to(torch.bfloat16))` (kayan nokta değilse dokunma); `unpack`: orijinal dtype'a geri çevir.

Bağlamı `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` diye tam adıyla çağır (kontrol bu çağrıyı izleyerek paketlenen baytı ölçüyor). Kontrol, saklanan baytın yarıya indiğini, ileri geçişin değişmediğini ve gradyanların yakın kaldığını ölçer.
