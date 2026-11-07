`F.cross_entropy`'yi kendin yaz, sonra hazırıyla karşılaştır.

- `ce_from_logits(Z, y)` — ortalama kayıp; `torch.logsumexp(Z, dim=1)` ve doğru sınıfın logit'i için `Z.gather(1, y[:, None])` ya da `Z[torch.arange(n), y]`. `F.cross_entropy`/`log_softmax`/`nn.CrossEntropyLoss` kullanma.
- `train_softmax(X, y, K, steps=300, lr=0.5)` → `nn.Linear(d, K)`; burada `nn.CrossEntropyLoss` serbest. Logit'leri doğrudan ver.

Kontrol, `ce_from_logits`'in autograd gradyanının $(P-Y)/n$ olduğunu ve $10^4$ büyüklüğünde logit'lerde sonlu kaldığını doğrular.
