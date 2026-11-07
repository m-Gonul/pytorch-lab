import time
import torch


def beam_loop(step, h0, sos, eos, k, T):
    B = h0.shape[0]
    toks_out, scores_out = [], []
    for b in range(B):
        beams = [([], 0.0, h0[b:b + 1], sos, False)]
        for _ in range(T):
            cand = []
            for seq, sc, h, last, done in beams:
                if done:
                    cand.append((seq + [eos], sc, h, eos, True))
                    continue
                lp, h2 = step(torch.tensor([last]), h)
                for v in range(lp.shape[1]):
                    cand.append((seq + [v], sc + float(lp[0, v]), h2, v, v == eos))
            cand.sort(key=lambda c: -c[1])
            beams = cand[:k]
        toks_out.append(beams[0][0])
        scores_out.append(beams[0][1])
    return torch.tensor(toks_out), torch.tensor(scores_out)


def beam_batched(step, h0, sos, eos, k, T):
    B, d = h0.shape
    h = h0.repeat_interleave(k, 0)
    last = torch.full((B * k,), sos)
    score = torch.full((B, k), float("-inf"))
    score[:, 0] = 0
    done = torch.zeros(B, k, dtype=torch.bool)
    hist = torch.zeros(B, k, 0, dtype=torch.long)
    for _ in range(T):
        lp, h2 = step(last, h)
        V = lp.shape[1]
        lp = lp.view(B, k, V)
        frozen = torch.full_like(lp, float("-inf"))
        frozen[..., eos] = 0
        lp = torch.where(done[..., None], frozen, lp)
        cand = (score[..., None] + lp).view(B, k * V)
        score, idx = cand.topk(k, -1)
        src, tok = idx // V, idx % V
        hist = torch.cat([hist.gather(1, src[..., None].expand(-1, -1, hist.shape[2])), tok[..., None]], 2)
        done = done.gather(1, src) | (tok == eos)
        h = h2.view(B, k, d).gather(1, src[..., None].expand(-1, -1, d)).view(B * k, d)
        last = tok.view(-1)
    return hist[:, 0], score[:, 0]
