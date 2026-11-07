from collections import Counter


def pair_counts(vocab):
    c = Counter()
    for w, f in vocab.items():
        for p in zip(w, w[1:]):
            c[p] += f
    return c


def merge_vocab(vocab, pair):
    out = {}
    for w, f in vocab.items():
        s, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                s.append(w[i] + w[i + 1]); i += 2
            else:
                s.append(w[i]); i += 1
        out[tuple(s)] = out.get(tuple(s), 0) + f
    return out


def train_bpe(word_freqs, num_merges):
    vocab = {tuple(w): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        c = pair_counts(vocab)
        if not c:
            break
        best = min(c, key=lambda p: (-c[p], p))
        merges.append(best)
        vocab = merge_vocab(vocab, best)
    return merges


def encode(word, merges):
    toks = list(word)
    for p in merges:
        toks = list(next(iter(merge_vocab({tuple(toks): 1}, p))))
    return toks


merges = train_bpe({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 4)
print(merges)                     # [('e','s'), ('es','t'), ('l','o'), ('lo','w')]
print(encode("lowest", merges))   # ['low', 'est']
