PyTorch. Görev: M4.13'teki tarih çevirisi (`dates`, `encode` verilir).

- `EncoderLayer(d, h, ff, dropout)` ve `DecoderLayer(d, h, ff, dropout)` — Post-LN, ReLU; **öznitelik adları PyTorch'unkilerle aynı** (`self_attn`, `multihead_attn`, `linear1`, `linear2`, `norm1..3`; `nn.MultiheadAttention(batch_first=True)` kullan) ki `nn.TransformerEncoderLayer`/`DecoderLayer` state_dict'i doğrudan yüklensin. `EncoderLayer.forward(x, pad)`; `DecoderLayer.forward(y, mem, causal, mem_pad)`.
- `Transformer(d=64, h=4, ff=128, N=2, dropout=0.1, emb_std=None)` — kaynak/hedef embedding'leri ($\times\sqrt d$) + sinüzoidal PE, N + N katman, çıktı doğrusal katmanı. `emb_std=None` → $d^{-1/2}$; bir sayı verilirse embedding'ler $N(0, \text{emb\_std}^2)$.
  - `encode(x)` → `(bellek, pad)`; `decode(yin, mem, pad)` → logit'ler; `forward(x, yin)`; `greedy(x, T=11)`.
