import math


def greedy_decode(step, sos, eos, max_len):
    seq, lp = [], 0.0
    for _ in range(max_len):
        probs = step(tuple([sos] + seq))
        tok = max(probs, key=probs.get)
        seq.append(tok)
        lp += math.log(probs[tok])
        if tok == eos:
            break
    return seq, lp


def beam_search(step, sos, eos, k, max_len, alpha=0.0):
    score = lambda s, lp: lp / (len(s) ** alpha if s else 1)
    beams, done = [([], 0.0)], []
    for _ in range(max_len):
        cand = []
        for seq, lp in beams:
            for tok, p in step(tuple([sos] + seq)).items():
                if p > 0:
                    cand.append((seq + [tok], lp + math.log(p)))
        cand.sort(key=lambda c: -c[1])
        beams = []
        for seq, lp in cand[:k]:
            (done if seq[-1] == eos else beams).append((seq, lp))
        if len(done) >= k or not beams:
            break
    pool = done if done else beams
    return sorted(pool, key=lambda c: -score(*c))


def pad_batch(seqs, pad=0):
    L = max(len(s) for s in seqs)
    return [s + [pad] * (L - len(s)) for s in seqs], [[True] * len(s) + [False] * (L - len(s)) for s in seqs]


def shift_right(y, sos):
    return [sos] + y[:-1]
