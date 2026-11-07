PyTorch. Görev: tarih biçimi çevirisi (Géron'un alıştırması) — `"April 22, 2019"`, `"22 April 2019"`, `"Apr 22 2019"` → `"2019-04-22"`. `dates(n, seed)` ve `encode(X, Y)` verilir (karakter düzeyinde; çıkış sözlüğü: PAD 0, SOS 1, EOS 2, rakamlar ve "-").

- `Seq2Seq(h=64, attn=True)` — embedding (32) + çift yönlü GRU kodlayıcı; başlangıç durumu $\tanh(W[\vec h_T;\overleftarrow h_1])$; `GRUCell` kod çözücü. `attn=True`: Bahdanau — $e=v^\top\tanh(W_1h_i+W_2s_{t-1})$, dolgu maskesi, bağlam GRU girdisine ve çıktı katmanına eklenir.
  - `forward(x, y)` — öğretmen zorlamasıyla logit'ler `(B, T, V)`.
  - `greedy(x, T=11)` — token'lar `(B, T)`.
  - `attention_map(x, T=11)` — açgözlü çözümleme sırasındaki ağırlıklar `(B, T, T_in)` (attn=True).
