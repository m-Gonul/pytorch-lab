- `attention(q, k, v, causal=False)` — `(…, T, dk)` girdiler; **elle** (`F.scaled_dot_product_attention` çağırmadan). Kontrol onunla karşılaştırır.
- `MultiHeadAttention(d, h)` — `nn.Module`:
  - `self.qkv = nn.Linear(d, 3 * d)` (çıktı sırası q, k, v), `self.proj = nn.Linear(d, d)`.
  - `forward(x, causal=False)` — `(B, T, d)` → `(B, T, d)`. Head'ler: `(B, T, h, dk)` → `transpose(1, 2)`.

Bu yerleşim `nn.MultiheadAttention(batch_first=True)`'ın `in_proj_weight`/`out_proj`'u ile birebir aynı; kontrol ağırlıkları kopyalayıp çıktıları karşılaştırır.
