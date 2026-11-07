NumPy. `toy_corpus(n_sent, seed)` verilir: insanlar (cinsiyet × statü × yaş) ve ülke–başkent–dil üçlüleriyle kurulmuş sentetik cümleler — her kelime, özelliklerine özgü bağlam kelimeleriyle birlikte geçer.

- `make_pairs(sentences, vocab, window)` → `(N, 2)` dizi (döngü serbest).
- `sgns_batch(Win, Wout, c, o, neg)` → `(toplam kayıp, gWin, gWout)`: `c, o (B,)`, `neg (B, k)`; gradyanlar tam şekilli dizilerde `np.add.at` ile birikir. Döngüsüz.
- `train_sgns(pairs, V, d=20, k=5, epochs=5, lr=0.05, batch=256, seed=0, counts=None)` → `Win` (doğrusal azalan lr, gürültü $\propto$ sayı$^{0.75}$).
- `analogy(E, vocab, a, b, c)` — kosinüs; $b-a+c$'ye en yakın, a, b, c hariç. Döngüsüz.
- `most_similar(E, vocab, w, n=3)`.
