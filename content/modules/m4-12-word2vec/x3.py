import math


def oov_hash(word):
    h = 0
    for ch in word:
        h = (31 * h + ord(ch)) % 2 ** 32
    return h


def char_ngrams(word, nmin=3, nmax=6):
    ...


def ngram_ids(word, buckets):
    ...


def word_vector(word, Z, buckets):
    ...


print(char_ngrams("where", 3, 3))
