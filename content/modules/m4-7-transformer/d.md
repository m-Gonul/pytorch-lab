nanoGPT tarzı:

- `GPT(vocab, block_size, d, n_layer, n_head)` — öznitelikler: `wte`, `wpe` (`nn.Embedding`), `blocks` (`nn.ModuleList`), `ln_f`, `head` (`nn.Linear(d, vocab, bias=False)`, ağırlığı `wte.weight` ile **aynı nesne**).
- Her blok: `ln1`, causal multi-head attention (`qkv` d→3d, `proj` d→d; `F.scaled_dot_product_attention(…, is_causal=True)` serbest), `ln2`, MLP (d→4d, `GELU(approximate="tanh")`, 4d→d). Tüm Linear'larda bias (head hariç).
- `forward(idx, targets=None)` → `(logits, loss)`; `targets` yoksa `loss = None`.

Kontrol parametre sayısını formülle, weight tying'i, nedenselliği (gelecek token değişince geçmiş logit'ler sabit) ve modelin kısa bir metni ezberleyebildiğini sınar.
