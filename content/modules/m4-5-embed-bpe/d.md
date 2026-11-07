Karakter düzeyinde bigram dil modeli: önceki karakter $a$'ya bakarak sonraki $b$'nin dağılımı. Sayım modeli $\hat p(b\mid a)=\frac{N_{ab}}{N_a}$; sinir ağı modeli `nn.Embedding(V, V)`: her $a$ için bir logit satırı.

- `BigramLM(V)` — `nn.Module`; `forward(idx)` → `(…, V)` logit.
- `train_bigram(ids, V, steps=500, lr=0.5)` → `(model, son_kayıp)`; girdiler `ids[:-1]`, hedefler `ids[1:]`, tam batch, `Adam`.
- `count_nll(ids, V)` → sayım modelinin ortalama NLL'i (gözlenen çiftler için $-\log\frac{N_{ab}}{N_a}$ ortalaması).

Kontrol, iki kaybın birbirine yakınsadığını doğrular.
