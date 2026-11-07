import numpy as np


def padding_waste(lengths, batches):
    lengths = np.asarray(lengths)
    real = sum(int(lengths[b].sum()) for b in batches)
    total = sum(int(lengths[b].max()) * len(b) for b in batches)
    return 1 - real / total


def random_batches(n, bs, rng):
    p = rng.permutation(n)
    return [p[i:i + bs] for i in range(0, n, bs)]


def bucket_batches(lengths, bs, rng, chunk=50):
    lengths = np.asarray(lengths)
    p = rng.permutation(len(lengths))
    out = []
    for i in range(0, len(p), chunk * bs):
        part = p[i:i + chunk * bs]
        part = part[np.argsort(lengths[part], kind="stable")]
        out += [part[j:j + bs] for j in range(0, len(part), bs)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]
