import math


def oov_hash(word):
    h = 0
    for ch in word:
        h = (31 * h + ord(ch)) % 2 ** 32
    return h


def char_ngrams(word, nmin=3, nmax=6):
    w = "<" + word + ">"
    out = [w[i:i + n] for n in range(nmin, nmax + 1) for i in range(len(w) - n + 1)]
    if w not in out:
        out.append(w)
    return out


def ngram_ids(word, buckets):
    return [oov_hash(g) % buckets for g in char_ngrams(word)]


def word_vector(word, Z, buckets):
    ids = ngram_ids(word, buckets)
    d = len(Z[0])
    return [sum(Z[i][k] for i in ids) / len(ids) for k in range(d)]


print(char_ngrams("where", 3, 3))
