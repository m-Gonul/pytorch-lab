def build_vocab(sentences, min_count=1):
    ...


def oov_hash(word):
    ...


def encode(tokens, vocab, n_oov):
    ...


def skipgram_pairs(ids, window):
    ...


def cbow_examples(ids, window):
    ...


def sgns_loss_grad(v, u_o, U_neg):
    ...


def noise_dist(counts, power=0.75):
    ...


def discard_prob(f, t=1e-5):
    ...


print(skipgram_pairs([0, 1, 2, 3, 4], 2))
