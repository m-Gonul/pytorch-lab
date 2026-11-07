Saf Python. Kelimeler sembol tuple'ı olarak: `("l", "o", "w")`.

- `pair_counts(vocab)` — `vocab: {tuple: frekans}` → `{(a, b): sayı}`.
- `merge_vocab(vocab, pair)` → yeni `vocab`; her kelimede soldan sağa, örtüşmeyen birleştirme.
- `train_bpe(word_freqs, num_merges)` — `word_freqs: {"low": 5, …}` → birleştirme listesi `[("e", "s"), ("es", "t"), …]`. Eşitlikte sözlük sırasında küçük çift (`min(çiftler, key=lambda p: (-sayı, p))`). Birleştirilecek çift kalmazsa dur.
- `encode(word, merges)` → token listesi.
