from collections import Counter


def pair_counts(vocab):
    ...


def merge_vocab(vocab, pair):
    ...


def train_bpe(word_freqs, num_merges):
    ...


def encode(word, merges):
    ...


merges = train_bpe({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 4)
print(merges)                     # [('e','s'), ('es','t'), ('l','o'), ('lo','w')]
print(encode("lowest", merges))   # ['low', 'est']
