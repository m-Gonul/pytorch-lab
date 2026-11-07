Model hazır (`TinyGPT`, d aşamasındakiyle aynı yapı). Sen yalnız artımlı yolu yazıyorsun:

- `forward_step(model, idx_new, cache, pos)` — `idx_new: (B, t)` yeni token'lar, `pos` bunların ilk mutlak konumu, `cache` her blok için `(K, V)` ya da `None` (`(B, h, S, dk)`). Döndür: `(logits (B, t, V), yeni_cache)`. Her blokta yeni k, v'yi cache'e `torch.cat(…, dim=2)` ile ekle. Maske: $i$. yeni sorgu (mutlak konum `pos + i`) `0 … pos + i` key'lerini görür → `attn_mask = arange(S)[None] <= (pos + arange(t))[:, None]`. Tek döngü: bloklar.
- `generate_kv(model, idx, n)` — greedy; önce prompt'un tamamıyla **prefill**, sonra her adımda yalnız son token. Tek döngü: adımlar.

Kontrol, her konumdaki logit'lerin tam ileri geçişle aynı olduğunu ve üretilen dizinin cache'siz greedy ile birebir aynı olduğunu doğrular.
