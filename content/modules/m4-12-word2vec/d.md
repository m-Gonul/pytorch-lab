PyTorch.

- `embedding_as_linear(emb)` — aynı hesabı yapan `nn.Linear(V, d, bias=False)`.
- `SGNS(V, d)` — `inp`, `out` `nn.Embedding`'ler (`sparse` parametreli); `forward(c, o, neg)` → ortalama SGNS kaybı (`F.logsigmoid`).
- `CBOW(V, d, pad)` — bağlam `(B, 2w)`, eksikler `pad` indeksiyle doldurulmuş (`padding_idx`); bağlam vektörlerinin ortalaması merkez kelimeyi negatif örneklemeyle tahmin eder.
- `train(model, batches, epochs, lr, sparse=False)` — `SparseAdam` (seyrek) ya da `Adam`.
