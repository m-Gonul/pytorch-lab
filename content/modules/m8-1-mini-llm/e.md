Profil çıkar ve hızlandır.

`LoopAttention` hazır: head başına ayrı `q`, `k`, `v` `nn.Linear`'ları ve head'ler üzerinde bir Python döngüsü. Sen yaz:

- `top_ops(fn, k=5)` — `torch.profiler.profile(activities=[ProfilerActivity.CPU])` altında `fn()`'i çalıştır; `key_averages()`'ı `self_cpu_time_total`'a göre sırala, ilk `k` işlemin adını döndür.
- `FusedAttention.from_loop(loop)` — tek bir `qkv = nn.Linear(d, 3d)`; ağırlıkları head'lerin ağırlıklarını **yığarak** kur (çıktı sırası: tüm q head'leri, tüm k head'leri, tüm v head'leri), aynı `proj`. `forward(x)`: tek matmul + reshape + `F.scaled_dot_product_attention(is_causal=True)`. **Head döngüsü yok.**

Kontrol çıktıların aynı olduğunu ve hız kazancını ölçer.
